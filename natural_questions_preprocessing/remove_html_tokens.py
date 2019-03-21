import jsonlines
import ast
import multiprocessing as mp
import json

def run(num_proc, num_file):
    count = 0
    new_dataset = []
    if len(str(num_file)) == 1:
        filename = "nq-train-0"+str(num_file)+".jsonl"
    if len(str(num_file)) == 2:
        filename = "nq-train-"+str(num_file)+".jsonl"
    fl = jsonlines.open(filename)
    for line in fl:
        new_sample = line_process(line)
        new_dataset.append(new_sample)

    out_filename = "dataset_clean_"+str(num_file)+".json"
    out = open(out_filename, 'w')
    json.dump(new_dataset, out)
    out.close()

def remove_html_tokens(document_tokens):
    answer_tokens = []
    p = 0
    t = 0
    l = 0
    subtract = 0
    subtracts = []
    for i in range(len(document_tokens)):
        tok = document_tokens[i]["token"]
        if not (tok.startswith('<') and tok.endswith('>')):
            answer_tokens.append(tok)
        if tok == "<P>":
            new_tok = "[Paragraph="+str(p)+"]"
            answer_tokens.append(new_tok)
            p += 1
        if tok == "<Ul>":
            new_tok = "[List="+str(l)+"]"
            answer_tokens.append(new_tok)
            l += 1
        if tok == "<Table>":
            new_tok = "[Table="+str(t)+"]"
            answer_tokens.append(new_tok)
            t += 1
        if tok.startswith('<') and tok.endswith('>') and tok not in ["<P>", "<Ul>", "<Table>"]:
            subtract += 1
        subtracts.append(subtract)

    return answer_tokens, subtracts

def count_spans(answer, subtracts):
    new_answer = {}
    if int(answer["start_token"]) > -1:
        new_answer["start_token"] = answer["start_token"] - subtracts[int(answer["start_token"])]
    if int(answer["start_token"]) == -1:
        new_answer["start_token"] = answer["start_token"]
    if int(answer["end_token"]) > -1:
        new_answer["end_token"] = answer["end_token"] - subtracts[int(answer["end_token"])]
    if int(answer["end_token"]) == -1:
        new_answer["end_token"] = answer["end_token"]

    return new_answer

def line_process(line):
    sample = ast.literal_eval(str(line))
    new_sample = {}
    new_sample["question_tokens"] = sample["question_tokens"]
    new_sample["question_text"] = sample["question_text"]
    long_answer_candidates = sample["long_answer_candidates"]
    document_tokens = sample["document_tokens"]
    
    answer_tokens, subtracts = remove_html_tokens(document_tokens)

    long_answer = sample["annotations"][0]["long_answer"]
    short_answers = sample["annotations"][0]["short_answers"]

    new_long_answer = count_spans(long_answer, subtracts)
    
    new_short_answers = []
    for j in range(len(short_answers)):
        new_short_answer = count_spans(short_answers[j], subtracts)
        new_short_answers.append(new_short_answer)

    new_long_answer_candidates = []
    for j in range(len(long_answer_candidates)):
        new_candidate = {}
        new_candidate["start_token"] = long_answer_candidates[j]["start_token"] - subtracts[int(long_answer_candidates[j]["start_token"])]
        new_candidate["end_token"] = long_answer_candidates[j]["end_token"] - subtracts[int(long_answer_candidates[j]["end_token"])]
        new_candidate["top_level"] = long_answer_candidates[j]["top_level"]
        new_long_answer_candidates.append(new_candidate)

    new_sample["long_answer"] = new_long_answer
    new_sample["short_answers"] = new_short_answers
    new_sample["long_answer_candidates"] = new_long_answer_candidates

    write_string = ' '.join(answer_tokens)
    f = write_string.find("References ( edit )")
    if f > -1:
        write_string = write_string[:f]

    new_sample["document_tokens"] = write_string

    return new_sample


workers = []
for jj in range(40):
    num_file = jj
    worker = mp.Process(target = run, args = (jj, num_file))
    workers.append(worker)
    worker.start()

for worker in workers:
    worker.join()
