from openai import OpenAI
import os
from langchain_openai import OpenAI
from langchain.vectorstores import Milvus
from langchain.docstore.document import Document
from langchain.embeddings import HuggingFaceInstructEmbeddings
import torch
import clip
from PIL import Image
import csv
import json
import jsonlines
import RPi.GPIO as GPIO
openai_api=""
dir_path=os.path.dirname(os.path.abspath(__file__))
output_txt="output.txt"
output = open(os.path.join(dir_path,output_txt),'w')
navigation_path=r"F:data\local_knowledge_base.json"
navigation_list=[]
with jsonlines.open(navigation_path,'r') as f:
    data = [obj for obj in f]
for data_ in data:
    navigation_list.append(data_['navigation_text'])
device = "cuda" if torch.cuda.is_available() else "cpu"
# Load Clip get label
model, preprocess = clip.load("ViT-B/32", device=device)
image_path=r"E:\Downloads\image\test.jpg"
label_list=[]
label_path=r"E:\Downloads\label\label.txt"
for line in open(label_path,'r',encoding='utf-8'):
    label_list.append(line.strip())
for label in label_list:
  print(label)
  print(type(label))
image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
text = clip.tokenize(label_list).to(device)
with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    logits_per_image, logits_per_text = model(image, text)
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()
print("Label probs:", probs)
for prob in probs.tolist():
  if prob is not None:
    max_score = max(prob)
    print("max_score:",max_score)
    max_index = prob.index(max_score)
    print("max_index:", max_index)
GPIO.setmode(GPIO.BOARD)
gpio_value_7 = GPIO.input(7)
gpio_value_8 = GPIO.input(8)
gpio_value_9 = GPIO.input(9)
gpio_value_10 = GPIO.input(10)
if gpio_value_7 == GPIO.HIGH | gpio_value_9 == GPIO.HIGH:
  camera_link=1
elif gpio_value_8 == GPIO.HIGH | gpio_value_10 == GPIO.HIGH:
  camera_link=0
if camera_link==0:
  if gpio_value_8 == GPIO.HIGH:
    query=label_list[max_index]+"is on your left"
  elif gpio_value_10 == GPIO.HIGH:
    query=label_list[max_index]+"is on your right"
elif camera_link==1:
  query=label_list[max_index]+"is in front of you"
else:
  print("query is error")
#加载embedding模型
EMBEDDING_MODEL_NAME = r"E:\Downloads\models\m3ebase\m3ebase"
embeddings = HuggingFaceInstructEmbeddings(
    model_name=EMBEDDING_MODEL_NAME,
    model_kwargs={"device": device},
)
docs_navigation = [Document(page_content=t,metadata={}) for t in navigation_list]
db_navigation = Milvus.from_documents(
    docs_navigation,
    embeddings,
    collection_name="db_navigation",
    connection_args={"host": "", "port": "19530"},
)
db_navigation.add_documents(docs_navigation)

docs_video_sim = db_navigation.similarity_search(query,k=1)
docs_video_sim_ = [doc.page_content for doc in docs_video_sim]
# docs_video_simi = "\n\n".join(docs_video_sim_)

PROMPT_TEMPLATE = """ 
You are ChatGPT, a large language model trained by OpenAI.
Knowledge cutoff: 2021-09
Current date: [2024-3-6]
你乘坐着一辆智能车辆，你希望它根据你的指示和观察行驶到目的地。 智能汽车应该如何行驶？ 请输出一系列动作:"
{context}",
根据对车辆周围环境的观察，目前你乘坐的自动驾驶车辆可以观察到："{question}"，
"""
llm = OpenAI(openai_api_key=openai_api)
text_=PROMPT_TEMPLATE.replace("{question}", query).replace("{context}", docs_video_sim_)
# print(llm.invoke(text_))
output.write("query:"+query+"\n"+"answer:"+llm.invoke(text_))