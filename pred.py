import os, csv, json
import argparse
import time
import requests
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import gensim
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from scipy.spatial.distance import cosine
import networkx as nx
from sentence_transformers import SentenceTransformer,CrossEncoder
from nltk.tokenize import sent_tokenize
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
nltk.download('punkt_tab')
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
from sklearn.metrics.pairwise import cosine_similarity
from joblib import Parallel, delayed

model_map = json.loads(open('/content/LongBench/config/model2path.json', encoding='utf-8').read())
maxlen_map = json.loads(open('/content/LongBench/config/model2maxlen.json', encoding='utf-8').read())

HF_API_URL = "hf_url" 
HF_API_TOKEN = "hf_token"

template_rag = open('/content/LongBench/prompts/0shot_rag.txt', encoding='utf-8').read()
template_no_context = open('/content/LongBench/prompts/0shot_no_context.txt', encoding='utf-8').read()
template_0shot = open('/content/LongBench/prompts/0shot.txt', encoding='utf-8').read()
template_0shot_cot = open('/content/LongBench/prompts/0shot_cot.txt', encoding='utf-8').read()
template_0shot_cot_ans = open('/content/LongBench/prompts/0shot_cot_ans.txt', encoding='utf-8').read()

def preprocess_text(text):
    text = text.lower()
    words = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    clean_text = " ".join(words)
    return clean_text

def generate_embeddings(sentences):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(sentences, convert_to_numpy=True)
    print("generated embeddings")
    return embeddings

def calculate_similarity_matrix(embeddings):
    sim_matrix = cosine_similarity(embeddings)
    np.fill_diagonal(sim_matrix, 0)
    print("calculated sim matrix")
    return sim_matrix

def clusterize_optimal(sim_matrix, num_clusters):
    dist_matrix = 1 - sim_matrix
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(dist_matrix)
    print("clustrized optimal")
    return clusters, kmeans.inertia_

def find_optimal_num_clusters(sim_matrix, max_clusters=20):
    def compute_sse(k):
        _, sse_value = clusterize_optimal(sim_matrix, k)
        return sse_value

    sse = Parallel(n_jobs=-1)(delayed(compute_sse)(k) for k in range(1, max_clusters + 1))

    sse_diff = np.diff(sse)
    sse_diff_diff = np.diff(sse_diff)

    optimal_num_clusters = np.argmin(sse_diff_diff) + 2

    print("found optimal num clusters")
    return optimal_num_clusters

def clusterize(sim_matrix, num_clusters):
    dist_matrix = 1 - sim_matrix
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(dist_matrix)
    print("clusterized")
    return clusters

def create_clusters_dict(tokenized_text, clusters):
  clusters_dict = {}
  for i, cluster in enumerate(clusters):
      if cluster not in clusters_dict:
          clusters_dict[cluster] = []
      clusters_dict[cluster].append((tokenized_text[i], i))
  return clusters_dict

def rank_clusters(clusters_dict, query):
    model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    cluster_pairs = []
    for cluster, cluster_sentences in clusters_dict.items():
        cluster_string = " ".join([sentence for sentence, _ in cluster_sentences])
        cluster_pairs.append((query, cluster_string)) # par (query, cluster_string)

    cluster_scores = model.predict(cluster_pairs)
    cluster_rank = sorted(zip(clusters_dict.keys(), cluster_scores), key=lambda x: x[1], reverse=True)
    print("ranked clusters")
    return cluster_rank

def select_top_sentences(tokenized_text, clusters, cluster_rank, query):
    model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    selected_sentences = []
    cluster_dict = {}

    for i, cluster in enumerate(clusters):
        if cluster not in cluster_dict:
            cluster_dict[cluster] = []
        cluster_dict[cluster].append(tokenized_text[i])

    min_score = min(score for _, score in cluster_rank)
    offset = abs(min_score) if min_score < 0 else 0

    relevancias_ajustadas = {cluster: (score + offset) for cluster, score in cluster_rank}
    total_relevancia = sum(relevancias_ajustadas.values())
    relevancias_normalizadas = {cluster: score / total_relevancia for cluster, score in relevancias_ajustadas.items()}

    for cluster, cluster_sentences in cluster_dict.items():
        relevancia_cluster = relevancias_normalizadas.get(cluster, 0)

        relevancia_ponderada = relevancia_cluster ** 0.6 #testar para saber qual o melhor valor

        top_k = max(1, int(relevancia_ponderada * len(cluster_sentences)))
        #top_k = max(1, int(0.2 * len(cluster_sentences)))

        query_cluster_pairs = [(query, sentence) for sentence in cluster_sentences]
        relevance_scores = model.predict(query_cluster_pairs)

        ranked_results = sorted(zip(cluster_sentences, relevance_scores), key=lambda x: x[1], reverse=True)[:top_k]

        selected_sentences.extend([sentence for sentence, score in ranked_results])
    print("selected top sentences")
    return selected_sentences

def format_compressed_sentences(ranked_sentences, tokenized_text):
    valid_indices = [i for i, sentence in enumerate(ranked_sentences) if 0 <= i < len(tokenized_text)]

    ordered_phrases = [tokenized_text[i] for i in valid_indices]
    compressed_prompt = ''.join(ordered_phrases)

    return compressed_prompt

def compress_prompt(prompt, query):
    context = prompt.strip()
    clean_text = preprocess_text(context)
    tokenized_text = sent_tokenize(clean_text)
    embeddings = generate_embeddings(tokenized_text)
    sim_matrix = calculate_similarity_matrix(embeddings)
    optimal_num_clusters = find_optimal_num_clusters(sim_matrix, max_clusters=20)
    clusters = clusterize(sim_matrix, optimal_num_clusters)
    clusters_dict = create_clusters_dict(tokenized_text, clusters)
    cluster_rank = rank_clusters(clusters_dict, query)
    selected_sentences = select_top_sentences(tokenized_text, clusters, cluster_rank, query)
    result = format_compressed_sentences(selected_sentences, tokenized_text)
    return result

def query(payload, headers):
	response = requests.post(HF_API_URL, headers=headers, json=payload)
	return response.json()

def query_llm(prompt, model, tokenizer, client=None, temperature=0.5, max_new_tokens=128, stop=None):
    max_len = maxlen_map[model]

    if model in model_map:
        input_ids = tokenizer.encode(prompt)
        if len(input_ids) > max_len:
            input_ids = input_ids[:max_len//2] + input_ids[-max_len//2:]
            prompt = tokenizer.decode(input_ids, skip_special_tokens=True)
    else:
        input_ids = tokenizer.encode(prompt, disallowed_special=())
        if len(input_ids) > max_len:
            input_ids = input_ids[:max_len//2] + input_ids[-max_len//2:]
            prompt = tokenizer.decode(input_ids)
    
    headers = {
        "Authorization": f"Bearer {HF_API_TOKEN}",
        "Content-Type": "application/json",
        "Accept": "application/json"
    }

    payload = {
        "inputs": prompt, 
        "parameters": {
            "temperature": temperature,
            "max_new_tokens": max_new_tokens
        }
    }

    tries = 0
    if model in model_map:
        model = model_map[model]
    while tries < 5:
        tries += 1
        try:
            output = query(payload, headers)
            return output[0]['generated_text']
        except requests.exceptions.RequestException as e:
            print(f"Erro na API ({e}), tentando novamente...")
            time.sleep(1)
    return ''

def extract_answer(response):
    response = response.replace('*', '')
    match = re.search(r'The correct answer is \(([A-D])\)', response)
    if match:
        return match.group(1)
    else:
        match = re.search(r'The correct answer is ([A-D])', response)
        if match:
            return match.group(1)
        else:
            return None


def get_pred(data, args, fout):
    model = args.model

    tokenizer = AutoTokenizer.from_pretrained(model_map[model], trust_remote_code=True)

    for item in tqdm(data):
        context = item['context']

        if args.rag > 0:
            template = template_rag
            retrieved = item["retrieved_context"][:args.rag]
            retrieved = sorted(retrieved, key=lambda x: x['c_idx'])
            context = '\n\n'.join([f"Retrieved chunk {idx+1}: {x['content']}" for idx, x in enumerate(retrieved)])
        elif args.no_context:
            template = template_no_context
        elif args.cot:
            template = template_0shot_cot
        else:
            template = template_0shot
        
        prompt_original = template.replace('$DOC$', context.strip())\
                                  .replace('$Q$', item['question'].strip())\
                                  .replace('$C_A$', item['choice_A'].strip())\
                                  .replace('$C_B$', item['choice_B'].strip())\
                                  .replace('$C_C$', item['choice_C'].strip())\
                                  .replace('$C_D$', item['choice_D'].strip())

        compressed_context = compress_prompt(context, item['question'])

        compressed_prompt = template.replace('$DOC$', compressed_context.strip())\
                                    .replace('$Q$', item['question'].strip())\
                                    .replace('$C_A$', item['choice_A'].strip())\
                                    .replace('$C_B$', item['choice_B'].strip())\
                                    .replace('$C_C$', item['choice_C'].strip())\
                                    .replace('$C_D$', item['choice_D'].strip())

        # gera resposta com prompt comprimido
        output_compressed = query_llm(compressed_prompt, model, tokenizer, client=None, temperature=0.1, max_new_tokens=128)
        if output_compressed == '':
            continue
        response_compressed = output_compressed.strip()
        pred_compressed = extract_answer(response_compressed)

        item['response_compressed'] = response_compressed
        item['pred_compressed'] = pred_compressed
        item['judge_compressed'] = pred_compressed == item['answer']
        item['prompt_length_compressed'] = len(compressed_prompt)

        # salva no arquivo
        fout.write(json.dumps(item, ensure_ascii=False) + '\n')
        fout.flush()
       

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", "-s", type=str, default="results")
    parser.add_argument("--model", "-m", type=str, default="Llama-3.1-8B-Instruct")
    parser.add_argument("--cot", "-cot", action='store_true') # set to True if using COT
    parser.add_argument("--no_context", "-nc", action='store_true') # set to True if using no context (directly measuring memorization)
    parser.add_argument("--rag", "-rag", type=int, default=0) # set to 0 if RAG is not used, otherwise set to N when using top-N retrieved context
    parser.add_argument("--n_proc", "-n", type=int, default=1)
    args = parser.parse_args([])
    
    os.makedirs(args.save_dir, exist_ok=True)
    print(args)

    if args.rag > 0:
        out_file = os.path.join(args.save_dir, args.model.split("/")[-1] + f"_rag_{str(args.rag)}.jsonl")
    elif args.no_context:
        out_file = os.path.join(args.save_dir, args.model.split("/")[-1] + "_no_context.jsonl")
    elif args.cot:
        out_file = os.path.join(args.save_dir, args.model.split("/")[-1] + "_cot.jsonl")
    else:
        out_file = os.path.join(args.save_dir, args.model.split("/")[-1] + ".jsonl")

    dataset = load_dataset('THUDM/LongBench-v2', split='train') # dataset = json.load(open('data.json', 'r', encoding='utf-8'))
    data_all = [{"_id": item["_id"], "domain": item["domain"], "sub_domain": item["sub_domain"], "difficulty": item["difficulty"], "length": item["length"], "question": item["question"], "choice_A": item["choice_A"], "choice_B": item["choice_B"], "choice_C": item["choice_C"], "choice_D": item["choice_D"], "answer": item["answer"], "context": item["context"]} for item in dataset if item["domain"] == "Single-Document QA"]

    has_data = {}
    if os.path.exists(out_file):
        with open(out_file, encoding='utf-8') as f:
            has_data = {json.loads(line)["_id"]: 0 for line in f}
    
    fout = open(out_file, 'a', encoding='utf-8')
    data = [item for item in data_all if item["_id"] not in has_data]

    data_subsets = [data[i::args.n_proc] for i in range(args.n_proc)]
    for subset in data_subsets:
      get_pred(subset, args, fout)

if __name__ == "__main__":
    main()

