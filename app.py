import gradio as gr
from youtube_transcript_api import YouTubeTranscriptApi
from transformers import AutoTokenizer
from transformers import pipeline
from transformers import AutoModelForQuestionAnswering
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch 

model_ckpt = "deepset/minilm-uncased-squad2"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
model = AutoModelForQuestionAnswering.from_pretrained(model_ckpt)
modelST = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

#input - video link, output - full transcript
def get_transcript(link):
  print("******** Inside get_transcript ********")
  print(f"link to be extracted is : {link}")
  video_id = link.split("=")[1]
  # Handle additional query parameters such as timestamp, ...
  video_id = video_id.split("&")[0]
  print(f"video id extracted is : {video_id}")
  transcript = YouTubeTranscriptApi.get_transcript(video_id)
  FinalTranscript = ' '.join([i['text'] for i in transcript])
  return FinalTranscript,transcript, video_id
  
  
#input - question and transcript, output - answer timestamp
def get_answers_timestamp(question, final_transcript, transcript):
  print("******** Inside get_answers_timestamp ********")

  context = final_transcript
  print(f"Input Question is : {question}")
  print(f"Type of trancript is : {type(context)}, Length of transcript is : {len(context)}")
  inputs = tokenizer(question, context, return_overflowing_tokens=True, max_length=512, stride = 25)

  #getting a list of contexts available after striding
  contx=[]
  for window in inputs["input_ids"]:
      #print(f"{tokenizer.decode(window)} \n")
      contx.append(tokenizer.decode(window).split('[SEP]')[1].strip())
  #print(ques)
  #print(contx)

  lst=[]
  pipe = pipeline("question-answering", model=model, tokenizer=tokenizer)
  for contexts in contx:
    lst.append(pipe(question=question, context=contexts))
  
  print(f"contx list is : {contx}")
  lst_scores = [dicts['score'] for dicts in lst] 
  print(f"lst_scores is : {lst_scores}")
  #getting highest and second highest scores
  idxmax = lst_scores.index(max(lst_scores))
  lst_scores.remove(max(lst_scores))
  idxmax2 = lst_scores.index(max(lst_scores))
  
  sentence_for_timestamp = lst[idxmax]['answer']
  sentence_for_timestamp_secondbest = lst[idxmax2]['answer']
  
  dftranscript = pd.DataFrame(transcript)

  embedding_1= modelST.encode(dftranscript.text, convert_to_tensor=True)
  embedding_2 = modelST.encode(sentence_for_timestamp, convert_to_tensor=True)
  embedding_3 = modelST.encode(sentence_for_timestamp_secondbest, convert_to_tensor=True)
  
  similarity_tensor = util.pytorch_cos_sim(embedding_1, embedding_2)
  idx = torch.argmax(similarity_tensor)
  start_timestamp = dftranscript.iloc[[int(idx)-3]].start.values[0]
  start_timestamp = round(start_timestamp)

  similarity_tensor_secondbest = util.pytorch_cos_sim(embedding_1, embedding_3)
  idx_secondbest = torch.argmax(similarity_tensor_secondbest)
  start_timestamp_secondbest = dftranscript.iloc[[int(idx_secondbest)-3]].start.values[0]
  start_timestamp_secondbest = round(start_timestamp_secondbest)

  return start_timestamp, start_timestamp_secondbest
   
    
def display_vid(url, question, sample_question=None, example_video=None):
  print("******** display_vid ********")
  if question == '':
    question = sample_question
  
  #get embedding and youtube link for initial video
  html_in = "<iframe width='560' height='315' src=" + url + " frameborder='0' allowfullscreen></iframe>"
  #print(html)
  
  if len(example_video) !=0 : #is not None:
    print(f"example_video is  : {example_video}")
    url = example_video[0]
  #get transcript
  final_transcript, transcript, video_id = get_transcript(url)
  
  #get answer timestamp
  #input - question and transcript, output - answer timestamp
  ans_timestamp, ans_timestamp_secondbest = get_answers_timestamp(question, final_transcript, transcript)
  
  #created embedding  width='560' height='315' 
  html_out = "<iframe width='730' height='400' src='https://www.youtube.com/embed/" + video_id + "?start=" + str(ans_timestamp) + "' title='YouTube video player' frameborder='0' allow='accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture' allowfullscreen></iframe>"
  print(f"html output is : {html_out}")
  html_out_secondbest = "<iframe width='730' height='400' src='https://www.youtube.com/embed/" + video_id + "?start=" + str(ans_timestamp_secondbest) + "' title='YouTube video player' frameborder='0' allow='accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture' allowfullscreen></iframe>"
  
  if question == '':
    print(f"Inside display_vid(), Sample_Question coming from Radio box is BEFORE : {sample_question}")
    sample_ques = set_example_question(sample_question)
    print(f"Inside display_vid(), Sample Question coming from Radio box is AFTER : {sample_ques}")
  else:
    sample_ques = question
  return html_out, html_out_secondbest, sample_ques, url

def set_example_question(sample_question):
    print(f"******* Inside Sample Questions ********")
    print(f"Sample Question coming from Radio box is : {sample_question}")
    print("What is the Return value : {gr.Radio.update(value=sample_question)}")
    return gr.Radio.update(value=sample_question) #input_ques.update(example)

demo = gr.Blocks()

with demo:
  gr.Markdown("<h1><center>Have you ever watched a lengthy video or podcast on YouTube and thought it would have been so much better if there had been 'explanatory' timestamps?</center></h1>")
  gr.Markdown(
        """### How many times have you seen a long video/podcast on Youtube and wondered only if there would have been 'explanatory' timestamps it would have been so much better..
            
        **Best part:** You don't even have to move away from the Space tab in your browser as the YouTube video gets played within the given View.
        """
    )
  with gr.Row():
    input_url = gr.Textbox(label="Input a Youtube video link") 
    input_ques = gr.Textbox(label="Ask a Question")

  with gr.Row():
    output_vid = gr.HTML(label="Video from timestamp 1", show_label=True)
    output_vid_secondbest = gr.HTML(label="Video from timestamp 2", show_label=True)
    
  with gr.Row():
    example_question = gr.Dropdown(
                    ["Choose a sample question", "Does video talk about different modalities", 
                    "does the model uses perceiver architecture?",
                    "when does the video talk about locked image tuning or lit?",
                    "comparison between gpt3 and jurassic?",
                    "Has flamingo passed turing test yet?",
                    "Any funny examples in video?",
                    "is it possible to download the stylegan model?",
                    "what was very cool?",
                    "what is the cool library?"], label= "Choose a sample Question", value=None)
  with gr.Row():
    example_video = gr.CheckboxGroup( ["https://www.youtube.com/watch?v=smUHQndcmOY"], label= "Choose a sample YouTube video") 
                                                                    
  b1 = gr.Button("Publish Video")
  
  b1.click(display_vid, inputs=[input_url, input_ques, example_question, example_video], outputs=[output_vid, output_vid_secondbest, input_ques, input_url])
  
  with gr.Row():
    gr.Markdown('''
    #### Model Credits
    1. [Question Answering](https://huggingface.co/deepset/minilm-uncased-squad2)
    1. [Sentence Transformer](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
    ''')
  
  with gr.Row(): 
    gr.Markdown("![visitor badge](https://visitor-badge.glitch.me/badge?page_id=gradio-blocks_ask_questions_to_youtube_videos)")

demo.launch(enable_queue=True, debug=True)