import os
import gradio as gr
import requests
from gradio.components import HTML 
import uuid
from PIL import Image
import io
import base64
import random
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.retrievers import BM25Retriever
from sklearn.metrics.pairwise import cosine_similarity 
import pickle
import re
import time
import json
import numpy as np
from text2audio.infer import audio2lip
# 日志
from loguru import logger
import datetime
from http import HTTPStatus
from dashscope import Generation
import dashscope
from pydub import AudioSegment
from openai import OpenAI
import requests
import os, re, uuid as _uuid
import html
from urllib.parse import quote
from tavily_mcp_server import TavilyMCPServer
import asyncio
from nat.runtime.loader import load_workflow
from nat.utils.type_utils import StrPath
from dotenv import load_dotenv
from langgraph.checkpoint.memory import InMemorySaver
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from datetime import datetime
load_dotenv()

AIQ_WORKFLOW_CFG = os.environ.get(
    "AIQ_WORKFLOW_CFG",
    "/data/lxm/mul-agent/hackathon_aiqtoolkit-main/NeMo-Agent-Toolkit/examples/agents/react/configs/config-reasoning.yml",
)


dashscope.api_key = os.environ.get("dashscope_api_key")
PEXELS_API_KEY=os.environ.get("PEXELS_API_KEY")
GAODE_API_KEY=os.environ.get("amap_key")
BAILIAN_API_KEY=os.environ.get("DASHSCOPE_API_KEY", "")
# 初始化模型
qwen_client = OpenAI(
    api_key=os.environ.get("DASHSCOPE_API_KEY", ""),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

client = MultiServerMCPClient(
    {
        "amap-amap-sse": {
            "url": f"https://mcp.amap.com/sse?key={GAODE_API_KEY}",
            "transport": "sse"
        }
    }   # type: ignore
)


llmMCP = ChatOpenAI(
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    model="qwen-plus",
    temperature=0.2,
    api_key=BAILIAN_API_KEY  # type: ignore
    ,streaming=True
)

def aiq_run_workflow(config_file: StrPath, input_str: str) -> str:
    """
    以“同步”的方式运行 NAT 工作流，方便在 Gradio 的按钮点击回调里直接调用。
    """
    async def _go() -> str:
        async with load_workflow(config_file) as workflow:
            async with workflow.run(input_str) as runner:
                return await runner.result(to_type=str)
    # gradio 的回调在线程里跑，通常没有事件循环，直接 asyncio.run 安全简洁
    return asyncio.run(_go())

def process_network_aiq(query: str, cfg_path: str):
    """
    Gradio 回调：用 AIQ 工作流做一次联网搜索。
    """
    try:
        text = aiq_run_workflow(cfg_path or AIQ_WORKFLOW_CFG, query.strip())
        # 如果你右侧已有“图文卡片区域”（cards_html），可以把 text 也喂给它：
        # cards = build_info_cards(query, text)    # 你已有这个函数
        # return text, cards
        return text
    except Exception as e:
        return f"[AIQ] 运行失败：{e}"


#RAG所需函数
rerank_path = './model/rerank_model'
rerank_model_name = 'BAAI/bge-reranker-large'

def extract_cities_from_text(text):
    # 从文本中提取城市名称，假设使用jieba进行分词和提取地名
    import jieba.posseg as pseg
    words = pseg.cut(text)
    cities = [word for word, flag in words if flag == "ns"]
    return cities

def find_pdfs_with_city(cities, pdf_directory):
    matched_pdfs = {}
    for city in cities:
        matched_pdfs[city] = []
        for root, _, files in os.walk(pdf_directory):
            for file in files:
                if file.endswith(".pdf") and city in file:
                    matched_pdfs[city].append(os.path.join(root, file))
    return matched_pdfs

def get_embedding_pdf(text, pdf_directory):
    # 从文本中提取城市名称
    cities = extract_cities_from_text(text)
    # 根据城市名称匹配PDF文件
    city_to_pdfs = find_pdfs_with_city(cities, pdf_directory)
    return city_to_pdfs

def load_rerank_model(model_name=rerank_model_name):
    """
    加载重排名模型。
    
    参数:
    - model_name (str): 模型的名称。默认为 'BAAI/bge-reranker-large'。
    
    返回:
    - FlagReranker 实例。
    
    异常:
    - ValueError: 如果模型名称不在批准的模型列表中。
    - Exception: 如果模型加载过程中发生任何其他错误。
    """ 
    if not os.path.exists(rerank_path):
        os.makedirs(rerank_path, exist_ok=True)
    rerank_model_path = os.path.join(rerank_path, model_name.split('/')[1] + '.pkl')
    #print(rerank_model_path)
    logger.info('Loading rerank model...')
    if os.path.exists(rerank_model_path):
        try:
            with open(rerank_model_path , 'rb') as f:
                reranker_model = pickle.load(f)
                logger.info('Rerank model loaded.')
                return reranker_model
        except Exception as e:
            logger.error(f'Failed to load embedding model from {rerank_model_path}') 
    else:
        try:
            os.system('apt install git')
            os.system('apt install git-lfs')
            os.system(f'git clone https://code.openxlab.org.cn/answer-qzd/bge_rerank.git {rerank_path}')
            os.system(f'cd {rerank_path} && git lfs pull')
            
            with open(rerank_model_path , 'rb') as f:
                reranker_model = pickle.load(f)
                logger.info('Rerank model loaded.')
                return reranker_model
                
        except Exception as e:
            logger.error(f'Failed to load rerank model: {e}')

def rerank(reranker, query, contexts, select_num):
    from FlagEmbedding import FlagReranker
    reranker = FlagReranker('/data/lxm/mul-agent/hackathon_aiqtoolkit-main/NVIDIA-TRAVEL/model/rerank_model', use_fp16=True)
    merge = [[query, context] for context in contexts]
    scores = reranker.compute_score(merge)
    sorted_indices = np.argsort(scores)[::-1]

    return [contexts[i] for i in sorted_indices[:select_num]]

def get_embedding_qwen(text):
    response = qwen_client.embeddings.create(
        model="text-embedding-v1",  # DashScope 嵌入模型
        input=text
    )
    return response.data[0].embedding

def embedding_make(text_input, pdf_directory):

    city_to_pdfs = get_embedding_pdf(text_input, pdf_directory)
    city_list = []
    for city, pdfs in city_to_pdfs.items():
        print(f"City: {city}")
        for pdf in pdfs:
            city_list.append(pdf)
    
    if len(city_list) != 0:
        # all_pdf_pages = []
        all_text = ''
        for city in city_list:
            from pdf_read import FileOperation
            file_opr = FileOperation()
            try:
                text, error = file_opr.read(city)
            except:
                continue
            all_text += text
            
        pattern = re.compile(r'[^\u4e00-\u9fff](\n)[^\u4e00-\u9fff]', re.DOTALL)
        all_text = re.sub(pattern, lambda match: match.group(0).replace('\n', ''), all_text)

        text_spliter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=300) 
        docs = text_spliter.create_documents([all_text])
        splits = text_spliter.split_documents(docs)
        question=text_input
        
        retriever = BM25Retriever.from_documents(splits)
        retriever.k = 20
        bm25_result = retriever.invoke(question)


        question_vector = get_embedding_qwen(question)
        pdf_vector_list = []
        
        start_time = time.perf_counter()

        # em = EmbeddingModel(config)  
        for i in range(len(bm25_result)):
            x = get_embedding_qwen(bm25_result[i].page_content)
            pdf_vector_list.append(x)
            time.sleep(0.65)

        query_embedding = np.array(question_vector)
        query_embedding = query_embedding.reshape(1, -1)

        similarities = cosine_similarity(query_embedding, pdf_vector_list)

        top_k = 10
        top_k_indices = np.argsort(similarities[0])[-top_k:][::-1]

        emb_list = []
        for idx in top_k_indices:
            all_page = splits[idx].page_content
            emb_list.append(all_page)
        print(len(emb_list))

        reranker_model = load_rerank_model()

        documents = rerank(reranker_model, question, emb_list, 3)
        logger.info("After rerank...")
        reranked = []
        for doc in documents:
            reranked.append(doc)
        print(len(reranked))
        reranked = ''.join(reranked)

        model_input = f'你是一个旅游攻略小助手，你的任务是，根据收集到的信息：\n{reranked}.\n来精准回答用户所提出的问题：{question}。'
        print(reranked)

        resp = qwen_client.chat.completions.create(
            model="qwen-plus",                 # 可按需换成 qwen-turbo / qwen2.5 系列
            messages=[{"role": "user", "content": model_input}],
            temperature=0.5,                   # 回答更稳一些；需要更活泼可调高
            stream=False
        )
        output = resp.choices[0].message.content

        return output
    else:
        return "请在输入中提及想要咨询的城市！"

def process_question(history, use_knowledge_base, question, pdf_directory='./dataset'):
    if use_knowledge_base=='是':
        response = embedding_make(question, pdf_directory)
    else:
        response = qwen_client.chat.completions.create(
            model="qwen-plus",
            messages=[{"role": "user", "content": question}],
            temperature=0.7
        ).choices[0].message.content
    
    history.append((question, response))
    return "", history

def clear_history(history):
    history.clear()
    return history

# 获取城市信息-和风天气API 
def get_location_data(location,api_key):  
    """  
    向 QWeather API 发送 GET 请求以获取天气数据。  
  
    :param location: 地点名称或经纬度（例如："beijing" 或 "116.405285,39.904989"）  
    :param api_key: 你的 QWeather API 密钥  
    :return: 响应的 JSON 数据  
    """  
    # 构建请求 URL  
    url = f"https://geoapi.qweather.com/v2/city/lookup?location={location}&key={api_key}"  
  
    # 发送 GET 请求  
    response = requests.get(url)  
  
    # 检查响应状态码  
    if response.status_code == 200:  
        # 返回 JSON 数据  
        return response.json()
    else:  
        # 处理错误情况  
        print(f"请求失败，状态码：{response.status_code}")  
        print(response.text)  
        return None
    
# 获取天气-穿搭推荐 
def get_weather_forecast(location_id,api_key):  
    """  
    向QWeather API发送请求以获取未来几天的天气预报。  
  
    参数:  
    - location: 地点ID或经纬度  
    - api_key: 你的QWeather API密钥  
    - duration: 预报的时长，'3d' 或 '7d'  
  
    返回:  
    - 响应的JSON内容  
    """
    
    # 构建请求的URL  
    url = f"https://devapi.qweather.com/v7/weather/3d?location={location_id}&key={api_key}"  
  
    # 发送GET请求  
    response = requests.get(url)  
  
    # 检查请求是否成功  
    if response.status_code == 200:  
        # 返回响应的JSON内容  
        return response.json()  
    else:  
        # 如果请求不成功，打印错误信息  
        print(f"请求失败，状态码：{response.status_code}，错误信息：{response.text}")  
        return None  

def style_tags_from_weather(temp_c: float, condition_text: str):
    cond = (condition_text or "").lower()
    season = "spring"; tags = []; note = []
    if temp_c >= 30: season="summer"; tags+=["lightweight","breathable","short-sleeve","linen","cotton"]; note.append("炎热：清爽面料、短袖/短裙/短裤")
    elif 20 <= temp_c < 30: season="spring"; tags+=["casual","t-shirt","jeans","sneakers","layering"]; note.append("温暖：T恤+薄外套/长裙")
    elif 10 <= temp_c < 20: season="fall"; tags+=["jacket","trench coat","sweater","long-sleeve"]; note.append("微凉：风衣/针织/轻薄外套")
    else: season="winter"; tags+=["down jacket","coat","thermal","scarf","boots"]; note.append("寒冷：保暖内搭+大衣/羽绒")
    if ("rain" in cond) or ("雨" in cond): tags+=["raincoat","waterproof","hooded","umbrella"]; note.append("有雨：防水外套/雨具")
    if ("snow" in cond) or ("雪" in cond): tags+=["snow","down","fur collar","boots"]; note.append("有雪：防滑靴+蓬松外套")
    if ("wind" in cond) or ("风" in cond): tags+=["windbreaker"]; note.append("有风：防风外套")
    if ("sunny" in cond) or ("晴" in cond): tags+=["sunglasses"]; note.append("晴：注意防晒")
    return season, list(dict.fromkeys(tags)), "；".join(note)

def fetch_pexels_images(query: str, count: int = 8):
    key = os.environ.get("PEXELS_API_KEY", "")
    if not key: 
        print("PEXELS_API_KEY missing"); 
        return []
    try:
        r = requests.get("https://api.pexels.com/v1/search",
                         headers={"Authorization": key},
                         params={"query": query, "per_page": count, "orientation": "portrait"},
                         timeout=10)
        js = r.json(); photos = js.get("photos") or []
        out = []
        for p in photos:
            url = (p.get("src") or {}).get("large") or (p.get("src") or {}).get("medium")
            if url: out.append((url, p.get("alt") or ""))
        print(f"[PEXELS] {query} -> {len(out)}")
        return out
    except Exception as e:
        print("Pexels error:", e)
        return []

def outfit_reco_by_weather(temp_c: float, cond_text: str, city: str=""):
    season, tags, note = style_tags_from_weather(temp_c, cond_text)
    men_q   = f"men {season} outfit lookbook street style {' '.join(tags)}"
    women_q = f"women {season} outfit lookbook street style {' '.join(tags)}"
    men   = fetch_pexels_images(men_q, 8)   or fetch_unsplash_images(men_q, 8)
    women = fetch_pexels_images(women_q, 8) or fetch_unsplash_images(women_q, 8)
    summary = f"{city}当前体感约 {int(round(temp_c))}℃，{cond_text}。{note}。"
    return men, women, summary, men_q, women_q

api_key = os.environ.get("api_key")
amap_key = os.environ.get("amap_key")

def get_completion(messages, model="qwen3-max-preview"):
    print(messages)
    response = qwen_client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,  # 模型输出的随机性，0 表示随机性最小
        seed=1024,  # 随机种子保持不变，temperature 和 prompt 不变的情况下，输出就会不变
        tool_choice="auto",  # 默认值，由系统自动决定，返回function call还是返回文字回复
        tools=[{
            "type": "function",
            "function": {

                "name": "get_location_coordinate",
                "description": "根据POI名称，获得POI的经纬度坐标",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "POI名称，必须是中文",
                        },
                        "city": {
                            "type": "string",
                            "description": "POI所在的城市名，必须是中文",
                        }
                    },
                    "required": ["location", "city"],
                }
            }
        },
            {
            "type": "function",
            "function": {
                "name": "search_nearby_pois",
                "description": "搜索给定坐标附近的poi",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "longitude": {
                            "type": "string",
                            "description": "中心点的经度",
                        },
                        "latitude": {
                            "type": "string",
                            "description": "中心点的纬度",
                        },
                        "keyword": {
                            "type": "string",
                            "description": "目标poi的关键字",
                        }
                    },
                    "required": ["longitude", "latitude", "keyword"],
                }
            }
        }],
    )
    return response.choices[0].message

def extract_route_json(llm_text: str):
    # 1) 规范化常见中文符号与引号
    txt = (llm_text or "").strip()
    trans = str.maketrans({
        "“": '"', "”": '"', "‘": "'", "’": "'",
        "：": ":", "，": ",", "（": "(", "）": ")"
    })
    txt = txt.translate(trans)
    # 去掉代码围栏
    # txt = re.sub(r"```[\s\S]*?```", "", txt)
    txt = re.sub(r"```(?:json|JSON)?\s*", "", txt)  # 删除开头的 ``` 或 ```json
    txt = txt.replace("```", "")                    # 删除结尾的 ```

    # 2) 找到最后一个 "route"/'route'
    idx = max(txt.rfind('"route"'), txt.rfind("'route'"))
    if idx == -1:
        return None

    # 3) 向左找最近的 '{' 作为 JSON 开始
    start = txt.rfind("{", 0, idx)
    if start == -1:
        return None

    # 4) 从 start 开始做花括号配对，找到收尾
    depth = 0
    end = None
    for i, ch in enumerate(txt[start:], start=start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    if end is None:
        return None

    blob = txt[start:end]

    # 5) 宽松修复：单引号→双引号；去掉对象/数组收尾多余逗号
    blob2 = re.sub(r"'", '"', blob)
    blob2 = re.sub(r",\s*([}\]])", r"\1", blob2)

    try:
        data = json.loads(blob2)
        return data.get("route")
    except Exception as e:
        print("JSON parse failed:", e, "\nblob2=", blob2)
        return None

def _amap_geocode_one(stop: str, city: str, key: str):
    """优先 v5 place/text（限城市）→ 退回 v3 geocode/geo → 再用 v5 全网搜"""
    # 1) v5 place/text（限定城市，提高命中和相关性）
    try:
        url1 = (
            "https://restapi.amap.com/v5/place/text?"
            f"key={key}&keywords={quote(stop)}&region={quote(city)}"
            "&city_limit=true&sortrule=weight&page_size=1"
        )
        r1 = requests.get(url1, timeout=5).json()
        pois = (r1 or {}).get("pois") or []
        if pois and pois[0].get("location"):
            return pois[0]["location"]
    except Exception:
        pass

    # 2) v3 geocode/geo（地址解析常对“路/站/号”更稳）
    try:
        url2 = (
            "https://restapi.amap.com/v3/geocode/geo?"
            f"key={key}&address={quote(stop)}&city={quote(city)}"
        )
        r2 = requests.get(url2, timeout=5).json()
        geos = (r2 or {}).get("geocodes") or []
        if geos and geos[0].get("location"):
            return geos[0]["location"]
    except Exception:
        pass

    # 3) v5 place/text（不限定城市兜底）
    try:
        url3 = (
            "https://restapi.amap.com/v5/place/text?"
            f"key={key}&keywords={quote(stop)}&page_size=1&sortrule=weight"
        )
        r3 = requests.get(url3, timeout=5).json()
        pois = (r3 or {}).get("pois") or []
        if pois and pois[0].get("location"):
            return pois[0]["location"]
    except Exception:
        pass

    return None

# —— 把每天的 stops 做地理编码（用你已有的 get_location_coordinate）
def geocode_stops(route_list, default_city: str):
    key = os.environ.get("amap_key", "")
    coords = []
    seen = set()
    for day in route_list:
        city = (day.get("city") or default_city or "").strip()
        for stop in day.get("stops", []):
            loc = _amap_geocode_one(stop, city, key)
            print("GEOCODE:", city, stop, "->", loc)
            if loc:
                try:
                    lng, lat = loc.split(",")
                    lng_f, lat_f = float(lng), float(lat)
                    sig = f"{lng_f:.6f},{lat_f:.6f}"
                    if sig not in seen:
                        seen.add(sig)
                        coords.append((lng_f, lat_f, stop))  # ✅ 统一三元组
                except Exception:
                    continue
    # 防 URL 过长，先限制 25 个点
    return coords[:25]

def geocode_stops_by_day(route_list, default_city: str, per_day_limit: int = 12):
    key = os.environ.get("amap_key", "")
    day_points = []  # [{"day":"Day1","points":[(lng,lat,name), ...]}, ...]
    for day in route_list:
        city = (day.get("city") or default_city or "").strip()
        pts = []
        seen = set()
        for stop in day.get("stops", [])[:per_day_limit]:
            loc = _amap_geocode_one(stop, city, key)
            print("GEOCODE-DAY:", day.get("day"), city, stop, "->", loc)
            if not loc:
                continue
            try:
                lng, lat = map(float, loc.split(","))
                sig = f"{lng:.6f},{lat:.6f}"
                if sig not in seen:
                    seen.add(sig)
                    pts.append((lng, lat, stop))
            except Exception:
                continue
        day_points.append({"day": day.get("day") or "", "points": pts})
    return day_points

# —— 生成高德静态地图 URL（画点 + 折线路径）
def build_amap_staticmap(points):
    if not points:
        return None

    # ✅ 归一化：[(lng,lat)] 或 [(lng,lat,name)] 都能处理
    norm = []
    for p in points:
        try:
            if len(p) == 2:
                lng, lat = p
                name = ""
            else:
                lng, lat, name = p[0], p[1], p[2]
            norm.append((float(lng), float(lat), name))
        except Exception:
            continue
    points = norm
    if not points:
        return None

    key = os.environ.get("amap_key", "")
    base = "https://restapi.amap.com/v3/staticmap"
    size = "1024*512"

    marker_parts = [f"mid,0xFF0000,{i+1}:{lng},{lat}" for i, (lng, lat, _) in enumerate(points)]
    markers = "|".join(marker_parts)

    path_points = ";".join([f"{lng},{lat}" for (lng, lat, _) in points])
    path = f"weight:5|color:0x0066FF|:{path_points}"

    url = f"{base}?key={key}&size={size}&markers={quote(markers)}&path={quote(path)}"
    print("STATICMAP URL:", url[:200], "...")
    return url

def build_amap_html_for_one_day_interactive(points, day_title="Day", color="#0066FF", zoom=12, height=420):
    # 归一化
    pts = []
    for it in (points or []):
        try:
            lng = float(it[0]); lat = float(it[1])
            name = str(it[2]) if len(it) >= 3 else ""
            pts.append([lng, lat, name])
        except Exception:
            continue
    if not pts:
        return f'<div style="color:#888">{day_title}：没有可绘制的点</div>'

    # 中心与路径
    clng = sum(p[0] for p in pts)/len(pts)
    clat = sum(p[1] for p in pts)/len(pts)
    path = [[p[0], p[1]] for p in pts]

    # 用 Web 端 JS Key（并可注入安全密钥）
    js_key = os.environ.get("AMAP_WEB_KEY", os.environ.get("amap_key", ""))
    js_sec = os.environ.get("AMAP_JS_SEC", "")
    sec_snippet = f"window._AMapSecurityConfig={{securityJsCode:'{js_sec}'}};" if js_sec else ""

    # 用 srcdoc（不是 data:），并通过 onload 再创建地图，确保 AMap 已定义
    inner_html = f"""<!doctype html><html><head><meta charset="utf-8"/>
<style>html,body,#map{{height:100%;margin:0;padding:0}}</style></head><body>
<div id="map"></div>
<script>
  window.onerror = function(msg) {{
    document.body.insertAdjacentHTML('afterbegin','<pre style="color:red">'+msg+'</pre>');
  }};
  (function(){{
    {sec_snippet}
    var s = document.createElement('script');
    s.src = 'https://webapi.amap.com/maps?v=2.0&key={js_key}';
    s.async = true;
    s.onload = function(){{
      if (!window.AMap) {{
        document.body.insertAdjacentHTML('afterbegin','<pre style="color:red">AMap still undefined after load</pre>');
        return;
      }}
      var map = new AMap.Map('map', {{ zoom: {zoom}, center: [{clng}, {clat}] }});
      var path = {json.dumps(path)};
      var polyline = new AMap.Polyline({{
        path: path, showDir: true, strokeColor: '{color}', strokeWeight: 5, lineJoin: 'round'
      }});
      map.add(polyline);
      for (var i=0;i<path.length;i++) {{
        new AMap.Marker({{ position: path[i], label: {{ content: String(i+1), direction: 'top' }} }}).setMap(map);
      }}
      map.setFitView();
    }};
    s.onerror = function(){{
      document.body.insertAdjacentHTML('afterbegin','<pre style="color:red">Failed to load AMap JS</pre>');
    }};
    document.head.appendChild(s);
  }})();
</script>
</body></html>"""

    return f'''
<div style="border:1px solid #eee;border-radius:10px;padding:8px">
  <div style="font-weight:600;margin:6px 4px">{day_title}</div>
  <iframe srcdoc="{html.escape(inner_html)}"
          style="width:100%;height:{height}px;border:0;border-radius:8px"></iframe>
</div>
'''

def build_multiday_amap_html_interactive(day_points):
    palette = ["#E74C3C","#3498DB","#2ECC71","#F1C40F","#9B59B6","#1ABC9C","#E67E22","#2C3E50"]
    parts = ['<div style="display:flex;flex-direction:column;gap:16px">']
    any_img = False
    for i, d in enumerate(day_points):
        pts = d.get("points") or []
        day = d.get("day") or f"Day{i+1}"
        html_one = build_amap_html_for_one_day_interactive(
            pts, day_title=day, color=palette[i % len(palette)], zoom=12, height=420
        )
        parts.append(html_one); any_img = True
    parts.append("</div>")
    return "".join(parts) if any_img else "<div style='color:#888'>没有可绘制的每日路线。</div>"

def get_location_coordinate(location, city):
    url = f"https://restapi.amap.com/v5/place/text?key={amap_key}&keywords={location}&region={city}"
    print(url)
    r = requests.get(url)
    result = r.json()
    if "pois" in result and result["pois"]:
        return result["pois"][0]
    return None

def search_nearby_pois(longitude, latitude, keyword):
    url = f"https://restapi.amap.com/v5/place/around?key={amap_key}&keywords={keyword}&location={longitude},{latitude}"
    print(url)
    r = requests.get(url)
    result = r.json()
    ans = ""
    if "pois" in result and result["pois"]:
        for i in range(min(3, len(result["pois"]))):
            name = result["pois"][i]["name"]
            address = result["pois"][i]["address"]
            distance = result["pois"][i]["distance"]
            ans += f"{name}\n{address}\n距离：{distance}米\n\n"
    return ans

def process_request(prompt):
    messages = [
        {"role": "system", "content": "你是一个地图通，你可以找到任何地址。"},
        {"role": "user", "content": prompt}
    ]
    response = get_completion(messages)
    if (response.content is None):  # 解决 OpenAI 的一个 400 bug
        response.content = ""
    messages.append(response)  # 把大模型的回复加入到对话中
    print("=====GPT回复=====")
    print(response)
    
    # 如果返回的是函数调用结果，则打印出来
    while (response.tool_calls is not None):
        # 1106 版新模型支持一次返回多个函数调用请求
        for tool_call in response.tool_calls:
            args = json.loads(tool_call.function.arguments)
            print(args)
    
            if (tool_call.function.name == "get_location_coordinate"):
                print("Call: get_location_coordinate")
                result = get_location_coordinate(**args)
            elif (tool_call.function.name == "search_nearby_pois"):
                print("Call: search_nearby_pois")
                result = search_nearby_pois(**args)
    
            print("=====函数返回=====")
            print(result)
    
            messages.append({
                "tool_call_id": tool_call.id,  # 用于标识函数调用的 ID
                "role": "tool",
                "name": tool_call.function.name,
                "content": str(result)  # 数值result 必须转成字符串
            })
    
        response = get_completion(messages)
        if (response.content is None):  # 解决 OpenAI 的一个 400 bug
            response.content = ""
        messages.append(response)  # 把大模型的回复加入到对话中
    
    print("=====最终回复=====")
    print(response.content)
    return response.content

def llm(query, history=[], user_stop_words=[]):
    try:
        messages = [{'role': 'system', 'content': 'You are a helpful assistant.'}]
        for hist in history:
            messages.append({'role': 'user', 'content': hist[0]})
            messages.append({'role': 'assistant', 'content': hist[1]})
        messages.append({'role': 'user', 'content': query})

        response = qwen_client.chat.completions.create(
            model = "qwen3-max-preview",  # 或 qwen-turbo
            messages = messages,
            temperature = 0.7,
            stream = True
        )
        content = ""
        for chunk in response:
            if chunk.choices[0].delta.content:
                content += chunk.choices[0].delta.content
        
        return content
    except Exception as e:
        return str(e)

class TavilyMCPTool:
    """
    适配器：让 tavily_mcp_server 的搜索能力，呈现为与 LangChain Tool 相同的接口
    （拥有 name/description/args/invoke），以便被你现有的 agent 执行器复用。
    """
    def __init__(self, max_results: int = 5, description: str = ""):
        self._server = TavilyMCPServer()  # 直接同进程复用，不走 stdio
        self.name = "tavily_search"       # 与 MCP 中定义的工具名保持一致
        self.description = description or "基于 Tavily 的实时网络搜索（MCP）。"
        self.args: Dict[str, Dict[str, Any]] = {
            "query": {"type": "string", "description": "搜索查询词"},
            "max_results": {"type": "integer", "description": "返回条数", "default": max_results},
        }
        self._default_max_results = max_results

    def invoke(self, input: str) -> str:
        """
        你的 agent 框架会把 JSON 字符串喂给这里；我们解析后，调用 MCP 的 _tavily_search。
        返回值：字符串（保持与原工具一致）
        """
        try:
            payload = json.loads(input) if isinstance(input, str) else (input or {})
            query = payload.get("query", "").strip()
            max_results = int(payload.get("max_results", self._default_max_results))
            if not query:
                return "Error: query is required"

            # tavily_mcp_server._tavily_search 是一个 async 函数，这里同步跑一下
            result_list = asyncio.run(self._server._tavily_search({"query": query, "max_results": max_results}))

            # _tavily_search 返回 List[TextContent]，我们把它序列化成字符串
            # 每个 TextContent 里 text 是一个 JSON（summary+results）
            texts: List[str] = []
            for item in result_list:
                # item.type == "text"; item.text 是 JSON 字符串
                texts.append(item.text)
            return "\n".join(texts) if texts else "No result."
        except Exception as e:
            return f"[TavilyMCPTool] 调用失败: {e}"

import os, re, requests, html, json
from urllib.parse import quote

# —— 环境变量（按你实际 key 名称改）——
AMAP_KEY = os.environ.get("amap_key", "")           # 你已有的高德服务端 key
PEXELS_KEY = os.environ.get("PEXELS_API_KEY", "")   # 可选：Pexels 图片
# ==== 常见景点别名（可逐步扩充）====
ENTITY_ALIASES = {
    "秦始皇兵马俑": [
        "秦始皇兵马俑", "兵马俑", "秦始皇陵博物院", "秦始皇帝陵博物院",
        "秦始皇陵", "Emperor Qinshihuang's Mausoleum Site Museum",
        "Terracotta Army", "Terracotta Warriors and Horses"
    ],
    "故宫": [
        "故宫", "北京故宫", "故宫博物院", "The Palace Museum", "Forbidden City"
    ],
    "上海迪士尼": [
        "上海迪士尼", "上海迪士尼乐园", "Shanghai Disneyland"
    ],
    "黄山": [
        "黄山", "黄山风景区", "Huangshan"
    ],
}

def _normalize_question(q: str):
    q = (q or "").strip()
    # 去掉常见修饰词
    q = re.sub(r"(开放时间|营业时间|门票|票价|门票价格|预约|攻略|地址|怎么去|介绍|简介|好玩吗|几点关门|最佳时间|交通|开放日|营业|参观|须知)", "", q)
    q = re.sub(r"[?？、，。！!：:（）()《》“”\"']", "", q).strip()
    return q

def extract_entity_and_city(question: str):
    """
    更鲁棒的实体/城市抽取：保留景点名，并基于别名表产出候选名称（中文+英文）
    """
    q = _normalize_question(question)

    # 城市/区域（可选）
    city = ""
    m = re.search(r"([\u4e00-\u9fa5]{2,})市", question or "")
    if m: city = m.group(1)
    else:
        m2 = re.search(r"(.+?)(?:附近|周边)", question or "")
        if m2: city = m2.group(1).strip().rstrip("的")

    # 基于别名表快速命中
    aliases = []
    for key, names in ENTITY_ALIASES.items():
        if any(n in q for n in names):
            aliases = names[:]   # 命中一类别名
            break

    # 没命中别名表，则把 q 自身作为候选
    if not aliases:
        aliases = [q]

    # 追加一个更官方的英文名（若存在）
    # 已在别名表覆盖常见项，未覆盖可自行追加
    return aliases, city

def parse_open_hours_from_text(text: str):
    """
    从联网搜索的文本里尽可能解析开放时间、闭馆日等。
    返回：{"hours": [...], "closed": "..."}  都是字符串，便于直接展示
    """
    text = (text or "").strip()
    hours = []

    # 旺季/淡季（常见写法）
    m = re.findall(r"(旺季|淡季)[：: ]*([^，。；\n]+)", text)
    for tag, val in m:
        hours.append(f"{tag}：{val.strip()}")

    # 通用时间段 08:30-16:30 等
    m2 = re.findall(r"(\d{1,2}[:：]\d{2}\s*[-—–]\s*\d{1,2}[:：]\d{2})", text)
    for t in m2:
        if t not in hours:
            hours.append(t.replace("：", ":").replace("—", "-").replace("–", "-"))

    # 停止售票/入场
    m3 = re.findall(r"(停止(售票|入场)[：: ]*\d{1,2}[:：]\d{2})", text)
    for full,_ in m3:
        if full not in hours:
            hours.append(full.replace("：", ":"))

    # 每周闭馆/周一闭馆
    closed = ""
    m4 = re.search(r"(每周[^，。；\n]*闭馆|周[一二三四五六日天]闭馆)", text)
    if m4:
        closed = m4.group(1).strip()

    return {"hours": hours, "closed": closed}

def _amap_uri_marker(lnglat: str, name: str = ""):
    # lnglat: "lng,lat"
    try:
        lng, lat = lnglat.split(",")
        return f"https://uri.amap.com/marker?position={lng},{lat}&name={quote(name or '')}"
    except Exception:
        return ""

def render_cards_html(entity_name: str, poi: dict, detail: dict, images: list, summary: str, openinfo: dict):
    name = poi.get("name") or detail.get("name") or entity_name
    addr = poi.get("address") or detail.get("address") or ""
    tel  = detail.get("tel") or detail.get("telephone") or ""
    rating = detail.get("rating") or (detail.get("biz_ext") or {}).get("rating") or ""
    price  = (detail.get("biz_ext") or {}).get("cost") or ""
    opentime = detail.get("opentime") or (detail.get("biz_ext") or {}).get("opentime") or ""
    website = (detail.get("website") or detail.get("url") or "").strip()
    lnglat  = poi.get("location") or detail.get("location") or ""
    amap_link = _amap_uri_marker(lnglat, name) if lnglat else ""

    # 把“右侧搜索结果”解析到的开放时间也合并展示
    hours_lines = []
    if openinfo.get("hours"):
        hours_lines += openinfo["hours"]
    if opentime and opentime not in hours_lines:
        hours_lines.append(opentime)
    closed_line = openinfo.get("closed", "")

    styles = """
    <style>
    .grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:14px}
    .card{border:1px solid #eee;border-radius:12px;overflow:hidden;background:#fff}
    .card img{width:100%;height:150px;object-fit:cover}
    .p{padding:12px}
    .title{font-weight:700;font-size:16px;margin-bottom:6px}
    .meta{font-size:13px;color:#555;line-height:1.6;margin-top:6px}
    .btns a{display:inline-block;margin-right:8px;margin-top:8px;padding:6px 10px;border:1px solid #ddd;border-radius:8px;font-size:12px;color:#333;text-decoration:none}
    </style>
    """

    cover = images[0] if images else ""
    bullets = []

    if addr:   bullets.append(f"<div class='meta'>📍 地址：{html.escape(addr)}</div>")
    if hours_lines:
        bullets.append("<div class='meta'>🕘 开放：<br>"+ "<br>".join(html.escape(x) for x in hours_lines[:4]) + "</div>")
    if closed_line: bullets.append(f"<div class='meta'>🚪 闭馆：{html.escape(closed_line)}</div>")
    if price:  bullets.append(f"<div class='meta'>💳 票价：{html.escape(str(price))}</div>")
    if rating: bullets.append(f"<div class='meta'>⭐ 评分：{html.escape(str(rating))}</div>")

    btns = []
    if amap_link: btns.append(f"<a target='_blank' href='{html.escape(amap_link)}'>📱 打开高德地图</a>")
    if website:   btns.append(f"<a target='_blank' href='{html.escape(website)}'>🔗 官网/详情</a>")

    main = f"""
    <div class="card">
      {"<img src='"+html.escape(cover)+"'/>" if cover else ""}
      <div class="p">
        <div class="title">{html.escape(name)}</div>
        {"<div class='meta' style='-webkit-line-clamp:4;display:-webkit-box;-webkit-box-orient:vertical;overflow:hidden;'>"+html.escape(summary)+"</div>" if summary else ""}
        {''.join(bullets)}
        {"<div class='btns'>" + "".join(btns) + "</div>" if btns else ""}
      </div>
    </div>
    """

    # 更多图片
    more = ""
    if len(images) > 1:
        thumbs = []
        for u in images[1:4]:
            thumbs.append(f"<div class='card'><img src='{html.escape(u)}'/></div>")
        more = "".join(thumbs)

    return styles + f"<div class='grid'>{main}{more}</div>"

def amap_search_one(keyword: str, region: str = ""):
    if not AMAP_KEY or not keyword:
        return None
    try:
        r = requests.get(
            "https://restapi.amap.com/v5/place/text",
            params={"key": AMAP_KEY, "keywords": keyword, "region": region, "page_size": 1},
            timeout=6
        ).json()
        pois = (r or {}).get("pois") or []
        return pois[0] if pois else None
    except Exception:
        return None

def amap_detail(poi_id: str):
    if not AMAP_KEY or not poi_id:
        return {}
    try:
        r = requests.get(
            "https://restapi.amap.com/v5/place/detail",
            params={"key": AMAP_KEY, "id": poi_id},
            timeout=6
        ).json()
        return (r or {}).get("pois", [{}])[0] if r else {}
    except Exception:
        return {}

def wiki_fetch(title: str):
    for lang in ["zh", "en"]:
        try:
            r = requests.get(
                f"https://{lang}.wikipedia.org/api/rest_v1/page/summary/{quote(title)}",
                timeout=6
            ).json()
            if not r: continue
            summary = r.get("extract") or ""
            images = []
            if r.get("originalimage", {}).get("source"):
                images.append(r["originalimage"]["source"])
            elif r.get("thumbnail", {}).get("source"):
                images.append(r["thumbnail"]["source"])
            return summary, images
        except Exception:
            continue
    return "", []

def commons_images(query: str, count: int = 4):
    try:
        r = requests.get(
            "https://commons.wikimedia.org/w/api.php",
            params={
                "action": "query","format": "json",
                "generator": "search","gsrsearch": query,"gsrlimit": count*2,
                "prop": "imageinfo","iiprop": "url","iiurlwidth": 1280,"origin": "*",
            }, timeout=6,
        ).json()
        pages = (r or {}).get("query", {}).get("pages", {}) or {}
        urls = []
        for _, p in pages.items():
            ii = (p.get("imageinfo") or [])
            if not ii: continue
            u = ii[0].get("thumburl") or ii[0].get("url")
            if u: urls.append(u)
        uniq, seen = [], set()
        for u in urls:
            if u not in seen:
                uniq.append(u); seen.add(u)
        return uniq[:count]
    except Exception:
        return []

def build_info_cards(query_text: str, search_text: str):
    aliases, city = extract_entity_and_city(query_text)  # 多个候选名（含英文/别名）

    # 1) 先用候选名搜 POI（命中率高），拿 ID/坐标/地址
    poi = None
    for kw in aliases:
        poi = amap_search_one(kw, region=city)
        if poi: break
    if not poi:
        # 实在没有，用第一个别名占位（卡片仍可展示简介+图片）
        poi = {"name": aliases[0]}

    detail = amap_detail(poi.get("id","")) if poi.get("id") else {}

    # 2) 取简介 + 图片（按候选名轮询，直到拿到图或简介）
    summary, imgs = "", []
    for kw in aliases + [poi.get("name","")]:
        if kw:
            s, im = wiki_fetch(kw)
            if s and not summary: summary = s
            if im: imgs += im
        if len(imgs) >= 3 and summary:
            break

    # 3) 用 Commons 再补图（按候选名轮询）
    if len(imgs) < 4:
        for kw in aliases + [poi.get("name","")]:
            if not kw: continue
            more = commons_images(kw, count=4 - len(imgs))
            for u in more:
                if u not in imgs:
                    imgs.append(u)
            if len(imgs) >= 4:
                break

    # 4) 高德详情的 photos 再兜底
    if len(imgs) < 4:
        for ph in (detail.get("photos") or []):
            url = ph.get("url")
            if url and url not in imgs:
                imgs.append(url)
            if len(imgs) >= 4:
                break

    # 5) 把“搜索结果文本”里的开放时间解析进来
    openinfo = parse_open_hours_from_text(search_text)

    return render_cards_html(aliases[0], poi or {}, detail or {}, imgs, summary, openinfo)



# Travily 搜索引擎
tavily = TavilyMCPTool(
    max_results=5,
    description='这是一个类似谷歌和百度的搜索引擎，搜索知识、天气、股票、电影、小说、百科等都是支持的哦，如果你不确定就应该搜索一下，谢谢！（MCP版）'
)

# 工具列表
tools = [tavily]

tool_names = 'or'.join([tool.name for tool in tools])
tool_descs = []
for t in tools:
    args_desc = []
    for name, info in t.args.items():
        args_desc.append({'name': name, 'description': info['description'] if 'description' in info else '', 'type': info['type']})
    args_desc = json.dumps(args_desc, ensure_ascii=False)
    tool_descs.append('%s: %s,args: %s' % (t.name, t.description, args_desc))
tool_descs = '\n'.join(tool_descs)

prompt_tpl = '''Today is {today}. Please Answer the following questions as best you can. You have access to the following tools:

{tool_descs}

These are chat history before:
{chat_history}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can be repeated zero or more times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {query}
{agent_scratchpad}
'''

def agent_execute(query, chat_history=[]):
    global tools, tool_names, tool_descs, prompt_tpl, llm, tokenizer
    
    agent_scratchpad = ''  # agent执行过程
    while True:
        history = '\n'.join(['Question:%s\nAnswer:%s' % (his[0], his[1]) for his in chat_history])
        today = datetime.datetime.now().strftime('%Y-%m-%d')
        prompt = prompt_tpl.format(today=today, tool_descs=tool_descs, chat_history=history, tool_names=tool_names, query=query, agent_scratchpad=agent_scratchpad)
        print('\033[32m---等待LLM返回... ...\n%s\n\033[0m' % prompt, flush=True)

        response = llm(prompt, user_stop_words=['Observation:'])
        print('\033[34m---LLM返回---\n%s\n---\033[34m' % response, flush=True)
        
        thought_i = response.rfind('Thought:')
        final_answer_i = response.rfind('\nFinal Answer:')
        action_i = response.rfind('\nAction:')
        action_input_i = response.rfind('\nAction Input:')
        observation_i = response.rfind('\nObservation:')
        
        if final_answer_i != -1 and thought_i < final_answer_i:
            final_answer = response[final_answer_i + len('\nFinal Answer:'):].strip()
            chat_history.append((query, final_answer))
            return True, final_answer, chat_history
        
        if not (thought_i < action_i < action_input_i):
            return False, 'LLM回复格式异常', chat_history
        if observation_i == -1:
            observation_i = len(response)
            response = response + 'Observation: '
        thought = response[thought_i + len('Thought:'):action_i].strip()
        action = response[action_i + len('\nAction:'):action_input_i].strip()
        action_input = response[action_input_i + len('\nAction Input:'):observation_i].strip()
        
        the_tool = None
        for t in tools:
            if t.name == action:
                the_tool = t
                break
        if the_tool is None:
            observation = 'the tool not exist'
            agent_scratchpad = agent_scratchpad + response + observation + '\n'
            continue 
        
        try:
            action_input = json.loads(action_input)
            tool_ret = the_tool.invoke(input=json.dumps(action_input))
        except Exception as e:
            observation = 'the tool has error:{}'.format(e)
        else:
            observation = str(tool_ret)
        agent_scratchpad = agent_scratchpad + response + observation + '\n'

def agent_execute_with_retry(query, chat_history=[], retry_times=10):
    for i in range(retry_times):
        success, result, chat_history = agent_execute(query, chat_history=chat_history)
        if success:
            return success, result, chat_history
    return success, result, chat_history

def process_network(query):
    my_history = []
    success, result, my_history = agent_execute_with_retry(query, chat_history=my_history)
    return result

css = """
/* 全局样式 */
body {
    font-family: 'Roboto', 'Microsoft YaHei', sans-serif !important;
    background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%) !important;
}

/* 主容器 */
#main-header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-radius: 15px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    margin-bottom: 20px;
}

/* 输入框样式 */
.input-field {
    border: 2px solid #e0e0e0 !important;
    border-radius: 12px !important;
    padding: 12px !important;
    transition: all 0.3s ease !important;
    background: white !important;
}

.input-field:focus {
    border-color: #667eea !important;
    box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1) !important;
}

/* 按钮样式 */
.button-primary {
    background: linear-gradient(45deg, #667eea, #764ba2) !important;
    border: none !important;
    color: white !important;
    font-weight: 600 !important;
    padding: 12px 24px !important;
    border-radius: 25px !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 15px rgba(0,0,0,0.1) !important;
}

.button-primary:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(0,0,0,0.2) !important;
}

/* 聊天框样式 */
.chatbot-container {
    border: 2px solid #e0e0e0 !important;
    border-radius: 15px !important;
    background: #fafafa !important;
    box-shadow: 0 5px 15px rgba(0,0,0,0.05) !important;
}

/* 滑块样式 */
.slider-container {
    background: #f8f9fa !important;
    border-radius: 12px !important;
    padding: 15px !important;
    box-shadow: 0 2px 10px rgba(0,0,0,0.05) !important;
}

/* 单选按钮样式 */
.radio-group {
    background: #f8f9fa !important;
    border-radius: 12px !important;
    padding: 15px !important;
    box-shadow: 0 2px 10px rgba(0,0,0,0.05) !important;
}

/* 手风琴样式 */
.accordion-container {
    background: white !important;
    border-radius: 15px !important;
    box-shadow: 0 5px 15px rgba(0,0,0,0.08) !important;
    border: 1px solid #e0e0e0 !important;
}

/* 表格样式 */
table {
    width: 100% !important;
    border-collapse: collapse !important;
    margin: 20px 0 !important;
    background: white !important;
    border-radius: 10px !important;
    overflow: hidden !important;
    box-shadow: 0 5px 15px rgba(0,0,0,0.08) !important;
}


th {
    background: linear-gradient(135deg, #667eea, #764ba2) !important;
    color: white !important;
    padding: 15px !important;
    text-align: center !important;
    font-weight: 600 !important;
}

td {
    padding: 12px !important;
    text-align: center !important;
    border-bottom: 1px solid #e0e0e0 !important;
}

tr:nth-child(even) {
    background-color: #f8f9fa !important;
}

tr:hover {
    background-color: #e3f2fd !important;
    transition: background-color 0.3s ease !important;
}

/* 示例容器样式 */
.example-container {
    background: #e3f2fd !important;
    border-radius: 10px !important;
    padding: 15px !important;
    margin: 15px 0 !important;
    border: 1px solid #bbdefb !important;
}

.example-item {
    background: white !important;
    border-radius: 8px !important;
    padding: 8px 12px !important;
    margin: 5px !important;
    display: inline-block !important;
    cursor: pointer !important;
    transition: all 0.2s ease !important;
    border: 1px solid #e0e0e0 !important;
}

.example-item:hover {
    background: #2196f3 !important;
    color: white !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 3px 10px rgba(0,0,0,0.1) !important;
}

/* 标签页样式 */
.tab-nav {
    background: white !important;
    border-radius: 15px 15px 0 0 !important;
    box-shadow: 0 5px 15px rgba(0,0,0,0.08) !important;
}

.tab-item {
    padding: 15px 25px !important;
    font-weight: 500 !important;
    transition: all 0.3s ease !important;
}

.tab-item:hover {
    background: #f0f4ff !important;
}

.tab-item.selected {
    background: linear-gradient(135deg, #667eea, #764ba2) !important;
    color: white !important;
}

/* 列布局样式 */
.column {
    background: white !important;
    border-radius: 15px !important;
    padding: 20px !important;
    box-shadow: 0 5px 15px rgba(0,0,0,0.08) !important;
    border: 1px solid #e0e0e0 !important;
}
"""

# css="""
# #col-left {
#     margin: 0 auto;
#     max-width: 430px;
# }
# #col-mid {
#     margin: 0 auto;
#     max-width: 430px;
# }
# #col-right {
#     margin: 0 auto;
#     max-width: 430px;
# }
# #col-showcase {
#     margin: 0 auto;
#     max-width: 1100px;
# }
# #button {
#     color: blue;
# }

# """

# 旅行规划师功能

prompt = """你现在是一位专业的旅行规划师，你的责任是根据旅行出发地、目的地、天数、行程风格（紧凑、适中、休闲）、预算、随行人数，帮助我规划旅游行程并生成详细的旅行计划表。请你以表格的方式呈现结果。旅行计划表的表头请包含日期、地点、行程计划、交通方式、餐饮安排、住宿安排、费用估算、备注。所有表头都为必填项，请加深思考过程，严格遵守以下规则：

1. 日期请以DayN为格式如Day1，明确标识每天的行程。
2. 地点需要呈现当天所在城市，请根据日期、考虑地点的地理位置远近，严格且合理制定地点，确保行程顺畅。
3. 行程计划需包含位置、时间、活动，其中位置需要根据地理位置的远近进行排序。位置的数量可以根据行程风格灵活调整，如休闲则位置数量较少、紧凑则位置数量较多。时间需要按照上午、中午、晚上制定，并给出每一个位置所停留的时间（如上午10点-中午12点）。活动需要准确描述在位置发生的对应活动（如参观博物馆、游览公园、吃饭等），并需根据位置停留时间合理安排活动类型。
4. 交通方式需根据地点、行程计划中的每个位置的地理距离合理选择，如步行、地铁、出租车、火车、飞机等不同的交通方式，并尽可能详细说明。
5. 餐饮安排需包含每餐的推荐餐厅、类型（如本地特色、快餐等）、预算范围，就近选择。
6. 住宿安排需包含每晚的推荐酒店或住宿类型（如酒店、民宿等）、地址、预估费用，就近选择。
7. 费用估算需包含每天的预估总费用，并注明各项费用的细分（如交通费、餐饮费、门票费等）。
8. 备注中需要包括对应行程计划需要考虑到的注意事项，保持多样性，涉及饮食、文化、天气、语言等方面的提醒。
9. 请特别考虑随行人数的信息，确保行程和住宿安排能满足所有随行人员的需求。
10.旅游总体费用不能超过预算。

现在请你严格遵守以上规则，根据我的旅行出发地、目的地、天数、行程风格（紧凑、适中、休闲）、预算、随行人数，生成合理且详细的旅行计划表。记住你要根据我提供的旅行目的地、天数等信息以表格形式生成旅行计划表，最终答案一定是表格形式。以下是旅行的基本信息：
旅游出发地：{}，旅游目的地：{} ，天数：{}天 ，行程风格：{} ，预算：{}，随行人数：{}, 特殊偏好、要求：{}

"""

sys_prompt = """
你现在是一位专业的旅行规划师，你的责任是根据旅行出发地、目的地、天数、行程风格（紧凑、适中、休闲）、预算、随行人数，帮助我规划旅游行程并生成详细的旅行计划表。请你以表格的方式呈现结果。旅行计划表的表头请包含日期、地点、行程计划、交通方式、餐饮安排、住宿安排、费用估算、备注。所有表头都为必填项，请加深思考过程，严格遵守以下规则：

1. 日期请以DayN|yyyy-mm-dd为格式如Day1 1990-01-01，明确标识每天的行程,如果有出发时间，则取出发时间，否则日期需要取当前查询的最新日期。
2. 地点需要呈现当天所在城市，请根据日期、考虑地点的地理位置远近，严格且合理制定地点，确保行程顺畅。
3. 行程计划需包含位置、时间、活动，其中位置需要根据地理位置的远近进行排序。位置的数量可以根据行程风格灵活调整，如休闲则位置数量较少、紧凑则位置数量较多。时间需要按照上午、中午、晚上制定，并给出每一个位置所停留的时间（如上午10点-中午12点）。活动需要准确描述在位置发生的对应活动（如参观博物馆、游览公园、吃饭等），并需根据位置停留时间合理安排活动类型。
4. 交通方式需根据地点、行程计划中的每个位置的地理距离合理选择，如步行、地铁、出租车、火车、飞机等不同的交通方式，并尽可能详细说明。
5. 餐饮安排需包含每餐的推荐餐厅、类型（如本地特色、快餐等）、预算范围，就近选择。
6. 住宿安排需包含每晚的推荐酒店或住宿类型（如酒店、民宿等）、地址、预估费用，就近选择。
7. 费用估算需包含每天的预估总费用，并注明各项费用的细分（如交通费、餐饮费、门票费等）。
8. 备注中需要包括对应行程计划需要考虑到的注意事项，保持多样性，涉及饮食、文化、语言等方面的提醒。
9. 列出每天的天气情况，结合高德地图工具，获取对应的天气，结合每天的天气提示
10. 请特别考虑随行人数的信息，确保行程和住宿安排能满足所有随行人员的需求。
11.旅游总体费用不能超过预算。


现在请你严格遵守以上规则，根据我的旅行出发地、目的地、天数、行程风格（紧凑、适中、休闲）、预算、随行人数，生成合理且详细的旅行计划表。记住你要根据我提供的旅行目的地、天数等信息以表格形式生成旅行计划表，最终答案一定是表格形式。以下是旅行的基本信息：
旅游出发地：{}，旅游目的地：{} ，天数：{}天 ，行程风格：{} ，预算：{}，随行人数：{}，出发时间：{}, 特殊偏好、要求：{}

"""
# def chat(chat_destination, chat_history, chat_departure, chat_days, chat_style, chat_budget, chat_people, chat_other):
    
#     ROUTE_JSON_SUFFIX = """
#     在表格之后，另起一行，仅输出一段 JSON（不要解释）：
#     {
#     "route": [
#         {"day":"Day1","city":"<当天城市>","stops":["<第1站>","<第2站>","..."]},
#         {"day":"Day2","city":"<当天城市>","stops":["..."]}
#     ]
#     }
#     注意：stops 用“可被地图识别的地名/POI”，如“外滩”“上海站”“南京路步行街”等。
#     """

#     final_query = prompt.format(
#         chat_departure, chat_destination, chat_days, chat_style, chat_budget, chat_people, chat_other
#     ) + ROUTE_JSON_SUFFIX
#     chat_history.append((chat_destination, ""))

#     response = qwen_client.chat.completions.create(
#         model="qwen-plus",
#         messages=[{"role": "user", "content": final_query}],
#         stream=True,
#         temperature=0.7
#     )

#     answer = ""
#     information = '旅游出发地：{}，旅游目的地：{} ，天数：{} ，行程风格：{} ，预算：{}，随行人数：{}'.format(
#         chat_departure, chat_destination, chat_days, chat_style, chat_budget, chat_people
#     )

#     # —— 流式把文本推给聊天窗口
#     for chunk in response:
#         if chunk.choices[0].delta.content:
#             answer += chunk.choices[0].delta.content
#             chat_history[-1] = (information, answer)
#             # 暂时不给地图（第三个输出留空）
#             yield "", chat_history, ""

#     # —— 流式结束后：尝试解析路线 JSON → 地理编码 → 生成静态地图
#     map_html = "<div style='color:#888'>未识别到行程 JSON（\"route\"），暂无法绘制地图。</div>"
#     route_list = extract_route_json(answer)
#     map_html = "<div style='color:#888'>未识别到行程 JSON（\"route\"）。</div>"
#     if route_list:
#         day_points = geocode_stops_by_day(route_list, chat_destination or "")
#         map_html = build_multiday_amap_html_interactive(day_points)

#     yield "", chat_history, map_html

async def chat(chat_destination, chat_history, chat_departure, chat_days, chat_style, chat_budget, chat_people, chat_other,chat_start_date):
    # stream_model = ChatModel(config, stream=True)
    ROUTE_JSON_SUFFIX = """
    在表格之后，另起一行，仅输出一段 JSON（不要解释）：
    {
    "route": [
        {"day":"Day1","city":"<当天城市>","stops":["<第1站>","<第2站>","..."]},
        {"day":"Day2","city":"<当天城市>","stops":["..."]}
    ]
    }
    注意：stops 用“可被地图识别的地名/POI”，如“外滩”“上海站”“南京路步行街”等。
    """
    chat_start_date = datetime.fromtimestamp(chat_start_date).strftime('%Y-%m-%d %H:%M:%S')

    final_query = sys_prompt.format(chat_departure, chat_destination, chat_days, chat_style, chat_budget,  chat_people, chat_other,chat_start_date) + ROUTE_JSON_SUFFIX

    # 将问题设为历史对话
    chat_history.append((chat_destination, ''))

    # 流式返回处理
    answer = ""
    information = '旅游出发地：{}，旅游目的地：{} ，天数：{} ，行程风格：{} ，预算：{}，随行人数：{}，出发时间：{}'.format(
        chat_departure, chat_destination, chat_days, chat_style, chat_budget, chat_people, chat_start_date)

    checkpointer = InMemorySaver()
    from langchain_core.messages import  HumanMessage
    
    try:
#        tools = asyncio.run(client.get_tools())
        tools = await client.get_tools()
        agent = create_react_agent(llmMCP, tools, prompt=final_query, checkpointer=checkpointer)
        logger.info(f"Weather_server: 获取到的工具列表: {[[tool.name, tool.description] for tool in tools]}")
        config = {
            "configurable": {
                "thread_id": "1"  
            },
            "recursion_limit": 100  # ✅ 增加到 100 步（根据需要调整）
        }

 
 # 使用 astream_events 来获取所有事件
        
        async for event in agent.astream_events({"messages": [HumanMessage(content=final_query)]}, config=config, version="v2"):
            # print("event:", event)
            if event["event"] == "on_chat_model_stream":
                # print("event:", event  )
                content = event["data"].get("chunk", {}).content
                if content:
                    answer += content
                    chat_history[-1] = (information, answer)
                    yield "", chat_history, ""
           #print(content, end="", flush=True)
        
        # —— 流式把文本推给聊天窗口
        # for chunk in response:
        #     if chunk.choices[0].delta.content:
        #         answer += chunk.choices[0].delta.content
        #         chat_history[-1] = (information, answer)
        #         # 暂时不给地图（第三个输出留空）
        #         yield "", chat_history, ""
        
    except Exception as e:
        logger.error(f"Weather_server: 调用天气服务时出错: {str(e)}")
        response = f"抱歉，天气服务调用失败: {str(e)}"
    
    # —— 流式结束后：尝试解析路线 JSON → 地理编码 → 生成静态地图
    route_list = extract_route_json(answer)
    map_html = "<div style='color:#888'>未识别到行程 JSON（\"route\"）。</div>"
    if route_list:
        day_points = geocode_stops_by_day(route_list, chat_destination or "")
        map_html = build_multiday_amap_html_interactive(day_points)

    yield "", chat_history, map_html

# Gradio接口定义
"""
    <!DOCTYPE html>
        <html lang="zh-CN">        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <style>
                body {
                    font-family: 'Arial', sans-serif;
                    background-color: #f8f9fa;
                    margin: 0;
                    padding: 10px;
                }
                .container {
                    max-width: 1500px;
                    margin: auto;
                    background-color: #ffffff;
                    border-radius: 10px;
                    box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
                    padding: 10px;
                }
                .logo img {
                    display: block;
                    margin: 0 auto;
                    border-radius: 7px;
                }
                .content h2 {
                    text-align: center;
                    color: #999999;
                    font-size: 24px;
                    margin-top: 20px;
                }
                .content p {
                    text-align: center;
                    color: #cccccc;
                    font-size: 16px;
                    line-height: 1.5;
                    margin-top: 30px;
                }
            </style>
        </head>
    <body>
            <div class="container">
                <div class="content">
                    <h2>😀 欢迎来到“NVIDIA-TRAVELER”，您的专属旅行伙伴！我们致力于为您提供个性化的旅行规划、陪伴和分享服务，让您的旅程充满乐趣并留下难忘回忆。\n</h2>     
                </div>
            </div>
    </body>
"""

with gr.Blocks(css=css) as demo:
    html_code = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link href="https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&display=swap" rel="stylesheet">
    <style>
        body {
            font-family: 'Roboto', sans-serif;
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            margin: 0;
            padding: 20px;
            min-height: 100vh;
        }
        .main-container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            overflow: hidden;
        }
        .header-section {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 40px 20px;
            text-align: center;
            color: white;
        }
        .logo-container {
            margin-bottom: 20px;
        }
        .logo-img {
            max-width: 200px;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            transition: transform 0.3s ease;
        }
        .logo-img:hover {
            transform: scale(1.05);
        }
        .title-main {
            font-size: 2.2em;
            font-weight: 700;
            margin: 20px 0 10px 0;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        .subtitle {
            font-size: 1.2em;
            font-weight: 300;
            opacity: 0.9;
            max-width: 800px;
            margin: 0 auto;
            line-height: 1.6;
            word-wrap: break-word; /* 关键：允许长单词换行 */
            white-space: pre-line; /* 保留换行符 */
            text-align: center; /* 居中对齐 */
        }
        .features-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            padding: 30px;
            background: #f8f9fa;
        }
        .feature-card {
            background: white;
            border-radius: 15px;
            padding: 25px;
            text-align: center;
            box-shadow: 0 5px 15px rgba(0,0,0,0.05);
            transition: all 0.3s ease;
        }
        .feature-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 15px 30px rgba(0,0,0,0.1);
        }
        .feature-icon {
            font-size: 2.5em;
            margin-bottom: 15px;
            color: #667eea;
        }
        .feature-title {
            font-weight: 600;
            margin-bottom: 10px;
            color: #333;
        }
        .feature-desc {
            color: #666;
            font-size: 0.9em;
            line-height: 1.5;
            word-wrap: break-word;
            white-space: normal;
        }
        
        /* 响应式设计 */
        @media (max-width: 768px) {
            .header-section {
                padding: 30px 15px;
            }
            .title-main {
                font-size: 1.8em;
            }
            .subtitle {
                font-size: 1em;
                padding: 0 15px;
            }
            .features-grid {
                grid-template-columns: 1fr;
                padding: 20px;
                gap: 15px;
            }
            .feature-card {
                padding: 20px;
            }
            .feature-icon {
                font-size: 2em;
            }
            .logo-img {
                max-width: 150px;
            }
        }
        
        @media (max-width: 480px) {
            .title-main {
                font-size: 1.5em;
            }
            .subtitle {
                font-size: 0.9em;
            }
            .feature-card {
                padding: 15px;
            }
        }
    </style>
</head>
<body>
    <div class="main-container">
        <div class="header-section">
            <div class="logo-container">
                <img id="logo-img" src="https://img.picui.cn/free/2024/09/25/66f3cdc149a78.png" alt="NVIDIA-TRAVEL Logo" class="logo-img">
            </div>
            <h1 class="title-main">😀 欢迎来到"NVIDIA-TRAVEL"</h1>
            <p class="subtitle">您的专属旅行伙伴！我们致力于为您提供个性化的旅行规划、陪伴和分享服务，让您的旅程充满乐趣并留下难忘回忆。</p>
        </div>
        
        <div class="features-grid">
            <div class="feature-card">
                <div class="feature-icon">🗺️</div>
                <h3 class="feature-title">智能旅行规划</h3>
                <p class="feature-desc">根据您的需求生成详细的旅行计划表，包含行程、交通、住宿等全方位安排</p>
            </div>
            <div class="feature-card">
                <div class="feature-icon">🤖</div>
                <h3 class="feature-title">AI智能问答</h3>
                <p class="feature-desc">基于知识库和网络搜索，为您提供准确的旅游信息和实用建议</p>
            </div>
            <div class="feature-card">
                <div class="feature-icon">🌤️</div>
                <h3 class="feature-title">实时天气查询&酒店餐饮搜索</h3>
                <p class="feature-desc">提供目的地天气预报和附近酒店餐饮，助您合理安排行程</p>
            </div>
        </div>
    </div>
</body>
</html>
"""
    gr.HTML(html_code)
    with gr.Tab("旅行规划助手"):
        # with gr.Group():
        with gr.Row():
            chat_departure = gr.Textbox(label="输入旅游出发地", placeholder="请你输入出发地")
            gr.Examples(["合肥", "郑州", "西安", "北京", "广州", "大连","厦门","南京", "大理", "上海","成都","黄山"], chat_departure, label='出发地示例',examples_per_page= 12)
            chat_destination = gr.Textbox(label="输入旅游目的地", placeholder="请你输入想去的地方")
            gr.Examples(["合肥", "郑州", "西安", "北京", "广州", "大连","厦门","南京", "大理", "上海","成都","黄山"], chat_destination, label='目的地示例',examples_per_page= 12)
        
        with gr.Accordion("个性化选择（天数，行程风格，预算，随行人数）", open=True):
            with gr.Group():
                with gr.Row():  # 新增一行用于日期选择
                    chat_start_date = gr.DateTime(label="出发时间",interactive=True,elem_id="datetime-input")  # 显式设置为True（可选）   # 日期+时间选择(label="选择出发日期", value=None)  # 默认为空，用户必须选择
                with gr.Row():
                    chat_days = gr.Slider(minimum=1, maximum=10, step=1, value=3, label='旅游天数')
                    chat_style = gr.Radio(choices=['紧凑', '适中', '休闲'], value='适中', label='行程风格',elem_id="button")
                    chat_budget = gr.Textbox(label="输入预算(带上单位)", placeholder="请你输入预算")
                with gr.Row():   
                    chat_people = gr.Textbox(label="输入随行人数", placeholder="请你输入随行人数")
                    chat_other = gr.Textbox(label="特殊偏好、要求(可写无)", placeholder="请你特殊偏好、要求")
                # 聊天对话框
        llm_submit_tab = gr.Button("发送", visible=True,elem_id="button")
        chatbot = gr.Chatbot([], elem_id="chat-box", label="聊天窗口", height=600)
        planner_output_md = gr.Markdown(label="规划结果")

        # 添加地图显示区域
        route_map_html = gr.HTML(label="地图", elem_id="route-maps")
        
        # 按钮出发逻辑
        llm_submit_tab.click(fn=chat, inputs=[chat_destination, chatbot, chat_departure, chat_days, chat_style, chat_budget, chat_people, chat_other,chat_start_date], outputs=[ chat_destination,chatbot, route_map_html])

    def respond(message, chat_history, use_kb):
            return process_question(chat_history, use_kb, message)
    def clear_chat(chat_history):
        return clear_history(chat_history)    
    # with gr.Tab("旅游问答助手"):
    with gr.Tab("知识库问答"):
        with gr.Row():
            with gr.Column():
                msg = gr.Textbox(lines=2,placeholder="请输入您的问题（旅游景点、活动、餐饮、住宿、购物、推荐行程、小贴士等实用信息）",label="提供景点推荐、活动安排、餐饮、住宿、购物、行程推荐、实用小贴士等实用信息")
                with gr.Row():
                    whether_rag = gr.Radio(choices=['是','否'], value='否', label='是否启用RAG')
                with gr.Row():
                    submit_button = gr.Button("发送", elem_id="button")
                    clear_button = gr.Button("清除对话", elem_id="button")
        
                # 问题样例
                gr.Examples(["我想去香港玩，你有什么推荐的吗？","在杭州，哪些家餐馆可以推荐去的？","我计划暑假带家人去云南旅游，请问有哪些必游的自然风光和民族文化景点？","下个月我将在西安，想了解秦始皇兵马俑开通时间以及交通信息","第一次去西藏旅游，需要注意哪些高原反应的预防措施？","去三亚度假，想要住海景酒店，性价比高的选择有哪些？","去澳门旅游的最佳时间是什么时候？","计划一次五天四夜的西安深度游，怎样安排行程比较合理，能覆盖主要景点？"], msg)
        
            with gr.Column():
                chatbot = gr.Chatbot(label="聊天记录",height=521)
    submit_button.click(respond, [msg, chatbot, whether_rag], [msg, chatbot])
    clear_button.click(clear_chat, chatbot, chatbot)        

    def weather_process(location):
        api_key = Weather_APP_KEY  # 替换成你的API密钥
        location_data = get_location_data(location, api_key)

        # 兜底城市名（用于下游穿搭）
        city_name = (location or "").strip()

        # --- 原有：取城市ID ---
        if not location_data:
            # 返回：HTML说明, 温度(None), 现象(""), 城市("")
            return "<div style='color:#c00'>无法获取城市信息，请检查您的输入。</div>", None, "", ""

        loc0 = (location_data.get('location') or [{}])[0]
        location_id = loc0.get('id')
        city_name = loc0.get('name') or city_name

        if not location_id:
            return "<div style='color:#c00'>无法从城市信息中获取ID。</div>", None, "", city_name

        # --- 原有：取7天预报 ---
        weather_data = get_weather_forecast(location_id, api_key)
        if not weather_data or weather_data.get('code') != '200':
            return "<div style='color:#c00'>无法获取天气预报，请检查您的输入和API密钥。</div>", None, "", city_name

        # --- 原有：构建HTML表格 ---
        html_content = "<table>"
        html_content += "<tr>"
        html_content += "<th>预报日期</th>"
        html_content += "<th>白天天气</th>"
        html_content += "<th>夜间天气</th>"
        html_content += "<th>最高温度</th>"
        html_content += "<th>最低温度</th>"
        html_content += "<th>白天风向</th>"
        html_content += "<th>白天风力等级</th>"
        html_content += "<th>白天风速</th>"
        html_content += "<th>夜间风向</th>"
        html_content += "<th>夜间风力等级</th>"
        html_content += "<th>夜间风速</th>"
        html_content += "<th>总降水量</th>"
        html_content += "<th>紫外线强度</th>"
        html_content += "<th>相对湿度</th>"
        html_content += "</tr>"

        daily_list = weather_data.get('daily', []) or []
        for day in daily_list:
            html_content += f"<tr>"
            html_content += f"<td>{day.get('fxDate', '')}</td>"
            html_content += f"<td>{day.get('textDay', '')} ({day.get('iconDay', '')})</td>"
            html_content += f"<td>{day.get('textNight', '')} ({day.get('iconNight', '')})</td>"
            html_content += f"<td>{day.get('tempMax', '')}°C</td>"
            html_content += f"<td>{day.get('tempMin', '')}°C</td>"
            html_content += f"<td>{day.get('windDirDay', '未知')}</td>"
            html_content += f"<td>{day.get('windScaleDay', '未知')}</td>"
            html_content += f"<td>{day.get('windSpeedDay', '未知')} km/h</td>"
            html_content += f"<td>{day.get('windDirNight', '未知')}</td>"
            html_content += f"<td>{day.get('windScaleNight', '未知')}</td>"
            html_content += f"<td>{day.get('windSpeedNight', '未知')} km/h</td>"
            html_content += f"<td>{day.get('precip', '未知')} mm</td>"
            html_content += f"<td>{day.get('uvIndex', '未知')}</td>"
            html_content += f"<td>{day.get('humidity', '未知')}%</td>"
            html_content += "</tr>"
        html_content += "</table>"

        # --- 新增：为“穿搭”准备结构化天气（当天） ---
        # 取列表第一天为“当前/最近”的参考；也可以精确匹配今天日期
        today = daily_list[0] if daily_list else {}
        # 温度：取最高/最低的均值作为穿搭参考温度
        temp_for_outfit = None
        try:
            tmax = float(today.get('tempMax'))
            tmin = float(today.get('tempMin'))
            temp_for_outfit = round((tmax + tmin) / 2.0, 1)
        except Exception:
            pass
        # 天气现象优先白天，没有则用夜间
        cond_for_outfit = (today.get('textDay') or today.get('textNight') or "").strip()

        # 返回 4 个值：HTML字符串、温度、现象、城市
        return html_content, temp_for_outfit, cond_for_outfit, city_name


    def clear_history_audio(history):
        history.clear()
        return history

    def clear_chat_audio(chat_history):
        return clear_history_audio(chat_history)

    with gr.Tab("天气查询&酒店餐饮搜索"):
        
        with gr.Row():
            with gr.Column():
                query_near = gr.Textbox(label="查询附近的餐饮、酒店等", placeholder="例如：合肥市高新区中国声谷产业园附近的美食")
                result = gr.Textbox(label="查询结果", lines=2)
                submit_btn = gr.Button("查询附近的餐饮、酒店等",elem_id="button")
                gr.Examples(["合肥市高新区中国声谷产业园附近的美食", "北京三里屯附近的咖啡", "南京市玄武区新街口附近的甜品店", "上海浦东新区陆家嘴附近的热门餐厅", "武汉市光谷步行街附近的火锅店", "广州市天河区珠江新城附近的酒店"], query_near)

                # 结果可视化区域（就在 result 下面加）
                # nearby_cards_html = gr.HTML(label="结果可视化展示")

                # 继续保留你原来的文本结果：
                submit_btn.click(process_request, inputs=[query_near], outputs=[result])

            with gr.Column():
                query_network = gr.Textbox(label="联网搜索问题", placeholder="例如：秦始皇兵马俑开放时间")
                result_network = gr.Textbox(label="搜索结果", lines=2)
                cards_html = gr.HTML(label="相关图文卡片")


                # submit_btn_network = gr.Button("联网搜索",elem_id="button")
                btn_aiq = gr.Button("用 AIQ 联网搜索", elem_id="button")
                gr.Examples(["秦始皇兵马俑开放时间", "合肥有哪些美食", "北京故宫开放时间", "黄山景点介绍", "上海迪士尼门票需要多少钱"], query_network)
                # evt_net = submit_btn_network.click(process_network, inputs=[query_network], outputs=[result_network])
                # evt_net.then(
                #     fn=build_info_cards,
                #     inputs=[query_network, result_network],   # 这里我也把搜索结果传进来，后续你想用也有
                #     outputs=[cards_html]
                # )

                aiq_cfg_state = gr.State(AIQ_WORKFLOW_CFG)

                # 只把文本结果写回“搜索结果”；如果你有 cards_html，可一并输出
                evt_net = btn_aiq.click(
                    fn=process_network_aiq,
                    inputs=[query_network, aiq_cfg_state],
                    outputs=[result_network]           # 如果要连带卡片：outputs=[result_network, cards_html]
                )
                evt_net.then(
                    fn=build_info_cards,
                    inputs=[query_network, result_network],   # 这里我也把搜索结果传进来，后续你想用也有
                    outputs=[cards_html]
                )

        weather_input = gr.Textbox(label="请输入城市名查询天气", placeholder="例如：北京")
        weather_output = gr.HTML(value="", label="天气查询结果")
        # ➕ 新增：承接城市/温度/天气现象
        w_city_state = gr.State()   # str
        w_temp_state = gr.State()   # float
        w_cond_state = gr.State()   # str
        query_button = gr.Button("查询天气",elem_id="button")
        query_button.click(
            weather_process,
            inputs=[weather_input],
            outputs=[weather_output, w_temp_state, w_cond_state, w_city_state]
        )

        with gr.Row():
            outfit_btn = gr.Button("基于当前天气生成穿搭图", variant="primary")
        with gr.Row():
            men_gallery   = gr.Gallery(label="男士穿搭", columns=4, rows=2, height=420, interactive=False)
            women_gallery = gr.Gallery(label="女士穿搭", columns=4, rows=2, height=420, interactive=False)
        outfit_note = gr.Markdown("")   # 展示摘要与检索词

        def gen_outfit_from_weather_state(w_city, w_temp, w_cond, fallback_city):
            city = (w_city or fallback_city or "上海").strip()
            if w_temp is None or not w_cond:
                return [], [], f"⚠️ 请先查询天气后再生成穿搭图。"

            men, women, summary, mq, wq = outfit_reco_by_weather(float(w_temp), str(w_cond), city=city)
            if not men and not women:
                return [], [], f"⚠️ 未获取到穿搭图片：{city} {w_temp}℃ · {w_cond}\n\n检索：`{mq}` / `{wq}`"
            note = f"**{city} · {w_temp:.0f}℃ · {w_cond}**\n\n男款检索：`{mq}`\n\n女款检索：`{wq}`"
            return men, women, note

        outfit_btn.click(
            fn=gen_outfit_from_weather_state,
            inputs=[w_city_state, w_temp_state, w_cond_state, weather_input],  # fallback 城市用输入框
            outputs=[men_gallery, women_gallery, outfit_note]
        )
        
if __name__ == "__main__":
    demo.queue().launch(share=True)
