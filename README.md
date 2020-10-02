# Collecting data about food safety events

Practical example from Human-in-the-Loop Machine Learning book

## Getting started

To run:

`python food_safety.py`


This will open a HTML window that will allow you to choose annotate four 
sequences in text-based reports:

- Hazard: Prohibited substance or ingredient
- Food:	Contaminated food or item
- Origin: Source country
- Destination: Destination country for the food


## Data

The data is approximately 30,000 public reports in the EU's
Food and Feed Safety Alerts (RASFF) portal .

They are all in English.

## Problem being addressed

Food Safety professionals want to collect data from incident reports 
about where pathogens or foreign objects have been detected in food.
- “I want to maintain a complete record of all recorded food safety incidents in the EU”
- “I want to track when different food safety incidents might have come from the same source”
- “I want to send warnings to specific countries when there are likely to be food safety incidents that have not yet been detected or reported”

## Annotation strategy

The interface allows the four fields to be typed in text inputs.

For all fields, the inputs have suggested auto-completes. 

The models are built using DistilBERT.

The system also continually retrains the model as new data is annotated. 

The interface ranks machine learning predictions for the fields first (if any),
and then backs-off to ngrams when no predictions fit the text being typed.

## Potential extensions

There are many different components in this architecture that could be extended or replaced. 
After playing around with the interface and looking at the results, think about what you might replace/change first.
(Numbers refer to book/chapter sections, but you don't need the book to experiment with this code.)

### Annotation Interface

- Predictive Annotations (See 11.5.4): Pre-populate the fields with the predictions, when the model is confident with that prediction. That will speed up annotation, but it can lead to more errors if the experts are primed to erroneously accept wrong predictions.  
- Adjudication (See 8.4, 11.5.4): Create a separate interface that allows the expert to quickly adjudicate examples that are high value to the model. This should be implemented as an optional additional strategy for the expert, not replacing their daily workflow.

### Annotation Quality Control

- Intra-annotator agreement (See 8.2): Domain experts often under-estimate their own consistency, so it might help to repeat some of the items for the same food safety expert will help measure whether this is the case.
- Predicting Errors (See 9.2.3): Build a model to explicitly predict where the expert is most likely to make errors, based on Ground-Truth Data, Inter/Intra-Annotator Agreement and/or the amount of time spent on each report (assuming that more time is spent on more complicated tasks). Then, use this model to flag where errors might occur and ask the expert to pay more attention and/or give those items to more people. 

### Machine Learning Architecture

- Synthetic Negative Examples (See 9.7): This dataset comes from templated text that is only about food safety events. This will make the model brittle in cases where the text is not about food safety events. For example, predicting that any word following “detected” is a pathogen. By asking the experts to make the minimal edits to create negative examples with existing contexts like “detected”, the model is less likely to erroneously learn the context. 
- Intermediate Task Training (See 9.4): If we can create a separate document-labeling model to predict “relevant” and “not relevant”, then we could use that model as a representation in the model predicting the fields. For example, if the expert already has a first step where they filter out the relevant from irrelevant reports, then that filtering step can become a predictive model in itself. It is likely that such a model will converge on features like pathogens and language about detection events. Therefore, it might improve the overall accuracy.

### Active Learning

- Re-ordering based on uncertainty (See 3.2-3.4): The system orders by date today. However, if the most uncertain items are ordered first instead, then that could improve the machine learning model faster and lead to a greater speed up overall. However, this will be an additional change to the current practice of the expert and is likely to feel slower initially as they tackle harder examples.
- Other Uncertainty Metrics (See 3.2): We use Ratio-of-Confidence as the basis for our confidence threshold because it seemed like the best fit for this problem. However, we can test empirically whether this is actually the case. 





