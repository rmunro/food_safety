import eel, os, random, sys, re
import time
import gzip
import csv
import hashlib
import json
import shutil
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
evaluation_data = []
labeled_data_path = "data/safety_reports_training.csv"
evaluation_data_path = "data/safety_reports_evaluation.csv"

labeled_ids = {} # track already-labeled data

predicted_confs = {} # predicted confidence of current label on current model by url
all_predicted_confs = {} # most recent predicted confidence, not necessarily from most recent model

verbose = True

@eel.expose
def save_report(report):
    '''Save annotated report 
    '''
    global label_indexes
    global labeled_data
    global labeled_ids
    global labeled_data_path
    global evaluation_data_path
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

    if is_evaluation(report_text):
        append_data(evaluation_data_path,[report])
        evaluation_data.append(report)
    else:
        append_data(labeled_data_path,[report])
        labeled_data.append(report)
        new_annotation_count += 1




def is_evaluation(text):
    hexval = hashlib.md5(text.encode('utf-8')).hexdigest()
    intval = int(hexval, 16)

    if intval%4 == 0:
        return True
    else:
        return False
   


def get_labels_from_text(text, annotation, label=1):
    '''Returns the per-token labels for annotation within text
       
    Note: returns the first match only
              
    '''
    tokens = tokenizer.tokenize(text)
    annotation_tokens = tokenizer.tokenize(annotation)
     
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
def get_next_report():
    '''Gets next report chronologically      
    '''
    global unlabeled_data
    for report in unlabeled_data:
        report_id = report[0]
        if report_id in labeled_ids:
            continue
        
        return report
    
    
@eel.expose
def get_candidate_spans(text, use_model_predictions=True, use_ngrams=True):
    '''Returns the potential spans in the text

    Uses all ngrams in the text if no model exists 
    
    Uses model predictions if a model exists 
       
    When a model exists, backs off to ngrams 
    by having them lower ranked than model predictions
              
    '''
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
        
    return spans
    



def get_ngrams(text, min_len=3,max_len=50):
    '''Returns word ngrams between given character lengths
    '''
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
    '''Loads already-annotated data
    '''
    if not os.path.exists(filepath):
        return []
    
    new_data = []
    with open(filepath, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for item in reader:
            labeled_ids[item[0]] = True # record this is now labeled
            new_data.append(item)

    return new_data



@eel.expose
def get_recent_reports(origin = "", hazard = "", food = ""):
    '''Loads reports in reverse chronological order
    
    Reports must match at least one of origin, hazard, or food
    '''
    global labeled_data
    global evaluation_data

    all_reports = labeled_data + evaluation_data
    
    all_reports.sort(reverse=True, key=lambda x: x[0]) 
    
    relevant_reports = []
    
    for report in all_reports:
        country_match = False
        if origin == report[5] and origin != "":
            country_match = True
        
        hazard_match = False
        if hazard == report[4] and hazard != "":
            hazard_match = True
        
        food_match = False
        if food == report[3] and food != "":
            food_match = True
        
        if country_match or hazard_match or food_match:
            relevant_reports.append(report)
    
    return relevant_reports   



def get_predictions(model, text):
    '''Get model predictions of potential spans within the text
    '''

    inputs = tokenizer(text, return_tensors="pt")
    
    candidates_by_label = {}
    
    with torch.no_grad():
        outputs = model(**inputs) #, labels=labels)  
         
        logits = outputs[0][0]

        tokens = [""] + tokenizer.tokenize(text) + [""]
        for ind in range(0, len(tokens)):
            token = tokens[ind]
            prob_dist = torch.softmax(logits[ind], dim=0)

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
            
            candidates = get_most_confident_spans(text, uncertainties, threshold=0.2)
            less_conf_candidates = get_most_confident_spans(text, uncertainties, threshold=0.6)
            for candidate in less_conf_candidates:
                if candidate not in candidates:
                    candidates.append(candidate)
                    
            candidates_by_label[label] = candidates
            
    return candidates_by_label
    
            
            
def convert_tokens_to_text(tokens, reference_text):
    '''Find best match for DistilBert tokens in the actual text
    '''
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



def get_most_confident_spans(text, uncertainties, threshold=0.5):
    '''Get all spans above the threshold in confidence 
    '''

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
    MODEL EVALUATION CODE HERE IF YOU WANT TO TEST THAT IT IS GETTING BETTER    
    '''
    
    current_model = new_model
    
    timestamp = re.sub('\.[0-9]*','_',str(datetime.now())).replace(" ", "_").replace("-", "").replace(":","")
    number_items = str(len(labeled_data))              
                     
    model_path = "models/"+timestamp+number_items+".model"
    current_model.save_pretrained(model_path)
    if verbose:
        print("saved model to "+model_path)
    clean_old_models()
    
        
    currently_training = False




def clean_old_models(max_prior=4):
     models = []
     
     files = os.listdir('models/') 
     for file_name in files:
         if os.path.isdir('models/'+file_name):
             if file_name.endswith(".model"):
                 models.append('models/'+file_name)
    
     if len(models) > max_prior:
         for filepath in models[:-4]:
             assert("models" in filepath and ".model" in filepath)
             if verbose:
                 print("removing old model "+filepath)
             shutil.rmtree(filepath)
    
    
    

def load_existing_model():
    global current_model 

    model_path = ""
    
    files = os.listdir('models') 
    for file_name in files:
        if file_name.endswith(".model"):
            model_path = 'models/'+file_name
                
    if model_path != '':    
        if verbose:
            print("Loading model from "+model_path)
        current_model = DistilBertForSequenceClassification.from_pretrained(model_path, num_labels=5)
        eel.sleep(0.1)
        # get_predictions()
    else:
        if verbose:
            print("Creating new uninitialized model (OK to ignore warnings)")

        current_model = DistilBertForTokenClassification.from_pretrained('distilbert-base-uncased', num_labels=5)



def check_to_train():
    global new_annotation_count
    global last_annotation
    global currently_training

    while True:
        print("Checking to retrain")
        
        if new_annotation_count > 0:
            retrain()
        
        # print(ct)
       
        eel.sleep(10) # Use eel.sleep(), not time.sleep()

eel.spawn(check_to_train)



# directories with data
unlabeled_file = gzip.open(unlabeled_data_path, mode='rt')
csvobj = csv.reader(unlabeled_file,delimiter = ',',quotechar='"')
for row in csvobj:
    unlabeled_data.append(row)

labeled_data = load_reports(labeled_data_path)
evaluation_data = load_reports(evaluation_data_path)


eel.start('food_safety.html', size=(800, 600))



