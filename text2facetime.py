from TTS.api import TTS
from transformers import pipeline, AutoTokenizer
import whisper
from scipy.io.wavfile import write
from playsound import playsound
import sounddevice as sd
import requests
import pandas as pd
import json
import re
import os
import threading
import asyncio
import cv2
import time
from Wav2lip_inference import main
import gradio as gr
from pydub import AudioSegment
from pydub.playback import play
import argparse
import multiprocessing
#loop = asyncio.new_event_loop()
# Load models (text to speech and speech to text and llm)
tts = TTS(model_name="tts_models/multilingual/multi-dataset/your_tts")
#model = whisper.load_model("base")
model_name = 'microsoft/phi-2'
generator = pipeline('text-generation', model=model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Take bio
freq = 16000
duration = 7
conv_hist = ''
user_input = 'new response'
response = ''

demo = "Ian's persona: Ian is a man that enjoys coding in python to learn more about machine learning. Ian works with machine learning by researching vision transformers, large language models and text to speech models. Ian's full name is Ian Codes. Ian is 22 years old. He is nice and likes alternative music. He is very knowledgeable about coding with python.\n<START>\n[DIALOGUE HISTORY]\nIan: Hello there I am Ian, I love coding and researching machine learning.\nYou: Hi how is it going Ian.\nIan: I'm doing well how about you?\nYou: I'm good too just learning more about local language models on youtube.\nIan: Oh that's cool I just have been using old models like gpt neo."

print('demo\n')
print(demo)

def perform_lip_sync(checkpoint_path, face, audio, outfile):
    args.checkpoint_path = checkpoint_path
    args.face = face
    args.audio = audio
    args.outfile = outfile
    args.resize_factor = 1
    args.rotate = False  
    args.crop = [0, -1, 0, -1]
    args.wav2lip_batch_size = 128
    args.resize_face = 1
    main(args)

if __name__ == '__main__':
    # Parse command-line arguments using argparse
    parser = argparse.ArgumentParser(description='Inference code to lip-sync videos in the wild using Wav2Lip models')
    parser.add_argument('--checkpoint_path', type=str, help='Name of saved checkpoint to load weights from', required=True)
    parser.add_argument('--face', type=str, help='Filepath of video/image that contains faces to use', required=True)
    parser.add_argument('--audio', type=str, help='Filepath of video/audio file to use as raw audio source', required=True)
    parser.add_argument('--outfile', type=str, help='Video path to save result', default='results/result_voice.mp4') 
    args = parser.parse_args()

    for i in range(0,10):
        num = str(i)
        newinput = input("Type response: ")
        user_input = "\nYou: " + newinput
        if re.search(r"\.$", user_input):
            user_input = user_input.replace(r".$", r"\.\\n")
        conv_hist += user_input + '\n'

        # Model response
        user_input = demo + conv_hist
        try:
            response = generator(user_input, max_new_tokens=25, do_sample=True, temperature=0.7, top_p=0.9,
                                 repetition_penalty=1.1)[0]['generated_text']
        except Exception as e:
            print("silly")
        response = response.replace(user_input, '')
        conv_hist += "Ian : " + response.strip('\n') + '\n'
        response = response.replace("You: ","")
        text_wav = response.replace("Ian: ", "")
        filepath = num + ".wav"
        user_input += response.strip('\n')

        # Text to speech
        print("Generating voice")
        tts.tts_to_file(text=text_wav, language="en", speaker_wav="iann.wav", file_path=filepath)
        print("playing audio:")
        #play(AudioSegment.from_file(filepath))

        # Perform lip-syncing
        face_filepath = "ian5sec25fps.mp4"
        checkpoint_filepath = "checkpoints/wav2lip_gan.pth"
        outfile = "results/result_voice.mp4"
        perform_lip_sync(checkpoint_filepath, face_filepath, filepath, outfile)
      
        # Update the Gradio interface after completing the loop
        outfile = "results/result_voice.mp4"  # Replace with actual outfile path
        passit = demo+conv_hist
        output_file_path = "passit_output.txt"

        # Write the content of the 'passit' variable to the text file
        with open(output_file_path, "w") as file:
            file.write(passit)