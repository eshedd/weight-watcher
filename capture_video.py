import numpy as np
import cv2
import argparse
import asyncio
import edge_tts
import subprocess
import torch
from train import ConvNet, CLASSES



TRAINER_VOICE = "en-GB-SoniaNeural"
SOUND_FILE = 'sound.mp3'

async def save_voice(reps) -> None:
    communicate = edge_tts.Communicate(str(reps+1), TRAINER_VOICE)
    await communicate.save(SOUND_FILE)

async def run(args):
   model = None
   if args.file:
      try:
         model = ConvNet(len(CLASSES))
         model.load_state_dict(torch.load(args.file))
         model.eval()
      except:
         print('No model found with path', args.file)


   cap = cv2.VideoCapture(0)
   if not cap.isOpened():
      print("Cannot open camera")
      exit()

   _, frame = cap.read()
   rows, cols, _ = map(int, frame.shape)
   prev = np.zeros((rows//4, cols//4))
   i, reps = 0, 0
   frames, preds = [], [-1]

   while True:
      if i % 60 == 0:
         # capture frame-by-frame
         success, frame = cap.read()
         # if frame is read correctly ret is True
         if not success:
            print("Can't receive frame (stream end?). Exiting ...")
            break

         # prettify frame
         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
         frame = cv2.pyrDown(cv2.pyrDown(frame))
         row, col = frame.shape
         frame = frame.flatten()
         frame = frame - np.mean(frame)
         frame /= np.std(frame)
         frame = np.reshape(frame, (row, col))

         # keep the dark stuff
         mask = frame>-1
         frame[mask] = 1

         # display difference frame
         diff = frame - prev
         cv2.imshow('frame', diff)

         prev = frame
         if model:
            crop = (diff.shape[1]-diff.shape[0])//2
            diff = diff[:, crop:-crop]
            pred = torch.argmax(model(torch.tensor(diff)[None,None,:,:].float())).item()

            # track reps
            if preds[-1] == 1 and pred == 0:
               reps += 1
               if reps % 10 == 0:
                  await save_voice(reps)
                  subprocess.call(["afplay", SOUND_FILE])

            preds.append(pred)
         
         frames.append(diff)


      if cv2.waitKey(1) == ord('q'): break
      i += 1


   # When everything done, release the capture
   cap.release()
   cv2.destroyAllWindows()

   print('recorded reps:', reps)


   if args.save:
      for i, frame in enumerate(frames):
         cv2.imwrite(f'./imgs/img{i}.jpg', frame*255)


   # display frames for review
   # if model:
   #    for i, frame in enumerate(frames):
   #       cv2.imshow(f"{i}:{CLASSES[preds[i+1]]}", frame)
   #       if cv2.waitKey(0) == ord('q'):
   #          break
   #       print(i, CLASSES[preds[i+1]])


if __name__ == '__main__':
   parser = argparse.ArgumentParser()
   parser.add_argument('-s', '--save', action=argparse.BooleanOptionalAction)
   parser.add_argument('-f', '--file', action='store', type=str, default=None)
   args = parser.parse_args()

   loop = asyncio.get_event_loop()
   loop.run_until_complete(run(args))
   loop.close()