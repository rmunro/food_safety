import eel, os, random, sys, re
import time
import gzip
import csv
import hashlib
import json
from datetime import datetime
from random import shuffle



from transformers import DistilBertTokenizer, DistilBertForTokenClassification
# downloading DistilBERT will take a while the first time

import torch
import torch.optim as optim


tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-cased', do_lower=False)
label_indexes = {"None":0, "Hazard":1, "Food":2, "Origin":3, "Destination":4}


eel.init('./')

current_model = None # current ML model
new_annotation_count = 1 # current count of annotations since last training (1 to trigger immediate training)
currently_training = False # if currently training


unlabeled_data = []
unlabeled_data_path = "safety_reports_v1.csv.gz"

labeled_data = []
labeled_data_path = "data/safety_reports_training.csv"
heldout_data_path = "data/safety_reports_heldout.csv"

labeled_ids = {} # track already-labeled data

predicted_confs = {} # predicted confidence of current label on current model by url
all_predicted_confs = {} # most recent predicted confidence, not necessarily from most recent model

verbose = True

@eel.expose
def save_report(report):
    global label_indexes
    global labeled_data
    global labeled_ids
    global labeled_data_path
    global heldout_data_path
    global new_annotation_count

    report_id = report[0]
    report_text = report[1]
    report_date = report[2]
    report_food = report[3]
    report_hazard = report[4]
    report_origin = report[5]
    report_destination = report[6]
    report.append(datetime.today().strftime('%Y-%m-%d-%H:%M:%S'))

    if report_text == "":
        return # empty submission
    
    hazard_labels = get_labels_from_text(report_text, report_hazard, label_indexes["Hazard"])
    food_labels = get_labels_from_text(report_text, report_food, label_indexes["Food"])
    origin_labels = get_labels_from_text(report_text, report_origin, label_indexes["Origin"])
    destination_labels = get_labels_from_text(report_text, report_destination, label_indexes["Destination"])
    
    print(hazard_labels)
    print(food_labels)
    print(origin_labels)
    print(destination_labels)
    
    text_labels = []
    numerical_labels = []
    for ind in range(0,len(hazard_labels)):    
        if hazard_labels[ind] != 0:
            text_labels.append("Hazard")
            numerical_labels.append(hazard_labels[ind])
        elif food_labels[ind] != 0:
            text_labels.append("Food")
            numerical_labels.append(food_labels[ind])
        elif origin_labels[ind] != 0:
            text_labels.append("Origin")
            numerical_labels.append(origin_labels[ind])
        elif destination_labels[ind] != 0:
            text_labels.append("Destination")
            numerical_labels.append(destination_labels[ind])
        else:
            text_labels.append("None")
            numerical_labels.append(0)

    tokens = tokenizer.tokenize(report_text)
    numerical_tokens = tokenizer.convert_tokens_to_ids(tokens)
    annotation = json.dumps([tokens, text_labels, numerical_tokens, numerical_labels])
    
    report.append(annotation)
    
    labeled_ids[report_id] = True

    if is_heldout(report_text):
        append_data(heldout_data_path,[report])
    else:
        append_data(labeled_data_path,[report])
        labeled_data.append(report)
        new_annotation_count += 1




def is_heldout(text):
    hexval = hashlib.md5(text.encode('utf-8')).hexdigest()
    intval = int(hexval, 16)

    if intval%4 == 0:
        return True
    else:
        return False
   

    
# get the per-token labels for annotation within text
# gets first match
def get_labels_from_text(text, annotation, label=1):
    tokens = tokenizer.tokenize(text)
    annotation_tokens = tokenizer.tokenize(annotation)
    print(tokens)
    print(annotation_tokens)
    
    if len(annotation_tokens) == 0:
        return [0] * len(tokens)

    labels = []    
    for ind in range(0, len(tokens)):
        if tokens[ind] == annotation_tokens[0]:
            cur_ind = ind
            matched = True
            for token in annotation_tokens:
                if tokens[cur_ind] != token:
                    matched = False
                    break
                cur_ind += 1
                
            if matched:
                for token in annotation_tokens:
                    labels.append(label)
                while len(labels) < len(tokens):
                    labels.append(0)
                break
        else:
            labels.append(0)
            
    return labels
    

@eel.expose
def pick_file(folder):
    if os.path.isdir(folder):
        return random.choice(os.listdir(folder))
    else:
        return 'Not valid folder'


@eel.expose
def get_next_report():
    global unlabeled_data
    for report in unlabeled_data:
        report_id = report[0]
        if report_id in labeled_ids:
            continue
        
        return report
    
    
@eel.expose
def get_candidate_spans(text, use_model_predictions=True, use_ngrams=True):
    global current_model
    global label_indexes 
    
    spans = {}
    # get potential spans. Back off to ngrams if none exist

    
    ngrams = get_ngrams(text)

    for label in label_indexes:
        if label == "None":
            continue
        spans[label] = []

    # MODEL PREDICTIONS GET TOP PRIORITY
    if use_model_predictions and current_model != None:
        spans = get_predictions(current_model, text)

    # NGRAMS WITHIN THE TEXT ARE NEXT PRIORITY
    if use_ngrams:
        for label in spans:
            candidates = spans[label]
            for ngram in ngrams:
                # TODO: use dict if needs to be faster
                if ngram not in candidates:
                    candidates.append(ngram)    
            spans[label] = candidates
  
    print(spans)
      
    return spans
    

def get_ngrams(text, min_len=3,max_len=50):
    print("ngrams from "+text)
    ngrams = []
    for start_ind in range(0, len(text)):
        if start_ind == 0 or (re.match("\W", text[start_ind-1]) and re.match("\w", text[start_ind])):
            # at start of token

            for end_ind in range(start_ind+(min_len-1), min(start_ind+(max_len-1),len(text))):
                if end_ind + 2 > len(text) or (re.match("\W", text[end_ind+1]) and re.match("\w", text[end_ind])):
                    string = text[start_ind: end_ind+1]                   
                    ngrams.append(string)

    return ngrams


def append_data(filepath, data):
    with open(filepath, 'a', errors='replace') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(data)
    csvfile.close()




def load_reports(filepath):
    # FOR ALREADY LABELED ONLY
    # csv format: [DATE, TEXT, URL,...]
    if not os.path.exists(filepath):
        return []
    
    with open(filepath, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for item in reader:
            labeled_ids[item[0]] = True # record this is now labeled
            labeled_data.append(item)



def get_predictions(model, text):
    inputs = tokenizer(text, return_tensors="pt")
    
    candidates_by_label = {}
    
    with torch.no_grad():
        outputs = model(**inputs) #, labels=labels)  
         
        logits = outputs[0][0]

        tokens = [""] + tokenizer.tokenize(text) + [""]
        for ind in range(0, len(tokens)):
            token = tokens[ind]
            prob_dist = torch.softmax(logits[ind], dim=0)
            # print(token)
            # print(prob_dist)

        for label in label_indexes:
            
            label_number = label_indexes[label]
            if label_number == 0:
                continue # skip non-spans
                
            uncertainties = []
                
            for ind in range(1, len(tokens)-1):
                prob_dist = torch.softmax(logits[ind], dim=0)
                conf = prob_dist[label_number]
                max_conf = torch.max(prob_dist)
                ratio_conf = 1 - conf/max_conf
                    
                uncertainties.append(ratio_conf.item())

            print(label+" "+str(uncertainties))
            
            candidates = get_most_confident_spans(text, uncertainties, threshold=0.2)
            less_conf_candidates = get_most_confident_spans(text, uncertainties, threshold=0.6)
            for candidate in less_conf_candidates:
                if candidate not in candidates:
                    candidates.append(candidate)
                    
            candidates_by_label[label] = candidates
            
    return candidates_by_label
    
            
            
def convert_tokens_to_text(tokens, reference_text):
    text = tokenizer.convert_tokens_to_string(tokens) 
    if text in reference_text:
        return text
    else:
        # try making spaces optional 
        regex = re.escape(text).replace("\ ","\s*")
        match = re.search("("+regex+")", reference_text)        
        if match:
            return match.group(1)
        else:
            # we couldn't find a match - return the tokenizer's text
            return text



# get best sequences
def get_most_confident_spans(text, uncertainties, threshold=0.5):
    tokens = tokenizer.tokenize(text)

    candidates = []
    
    for start_ind in range(0, len(tokens)):
        if uncertainties[start_ind] <= threshold and not tokens[start_ind].startswith('##'):
            for end_ind in range(start_ind, len(tokens)):
                if uncertainties[end_ind] <= threshold  and not tokens[end_ind].startswith('##'):                   
                    candidate = convert_tokens_to_text(tokens[start_ind:end_ind+1], text)                    
                    if candidate not in text:
                        print("WARNING couldn't find span in text: "+candidate)
                    else:
                        candidates.append(candidate)
                        
    # return in descending length 
    candidates.sort(key=len, reverse=True)
    
    return candidates



def train_item(model, annotations, text):
    model.zero_grad() 
    
    # numerical labels, padded for sentence start/fin tokens
    numerical_labels = [0] + annotations[3] + [0]        
        
    labels = torch.tensor(numerical_labels).unsqueeze(0) 
    inputs = tokenizer(text, return_tensors="pt")
            
    outputs = model(**inputs, labels=labels)                
    loss, logits = outputs[:2]   
    
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    loss.backward()
    optimizer.step()     



def retrain(epochs_per_item=2, min_to_train=5):
    global current_model
    global currently_training
    global new_annotation_count
    global labeled_data
    

    if currently_training:
        "skipping while model already training"
        return
    
    if len(labeled_data) < min_to_train:
        print("too few annotations to train: "+str(len(labeled_data)))        
        return
        
    currently_training = True
    new_annotation_count = 0
    
    new_model = DistilBertForTokenClassification.from_pretrained('distilbert-base-cased', num_labels=5)

        
    for epoch in range(0, epochs_per_item):
        print("epoch "+str(epoch))
        shuffle(labeled_data) 
        for report in labeled_data:
            annotations = json.loads(report[8])
            report_text = report[1]
            train_item(new_model, annotations, report_text)
         

            eel.sleep(0.01) # allow other processes through
 
    '''
    new_fscore = evaluate_model(new_model)
    current_fscore = evaluate_model(current_model)
      
    if(new_fscore > current_fscore):
        print("replacing model!")
        current_model = new_model
        get_uncertainty()
    else:
        print("staying with old model")
    '''
    
    current_model = new_model
        
    currently_training = False




def check_to_train():
    global new_annotation_count
    global last_annotation
    global currently_training

    while True:
        print("Checking to retrain")
        
        if new_annotation_count > 0:
            retrain()
        
        # print(ct)
       
        eel.sleep(10)                  # Use eel.sleep(), not time.sleep()

eel.spawn(check_to_train)



# directories with data
unlabeled_file = gzip.open(unlabeled_data_path, mode='rt')
csvobj = csv.reader(unlabeled_file,delimiter = ',',quotechar='"')
for row in csvobj:
    unlabeled_data.append(row)

load_reports(labeled_data_path)


eel.start('food_safety.html', size=(800, 600))



