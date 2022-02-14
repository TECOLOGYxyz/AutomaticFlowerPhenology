# -*- coding: utf-8 -*-
"""
Evaluating ML performance - SingleClass

@author: Hjalte Mann, AU
hjaltemrm@gmail.com
TECOLOGYxyz
"""
import pandas as pd

iou_threshold = 0.5

# Get the ground truth annotations on the format [filename, x_min, y_min, x_max, y_max, id_gt]
gt = pd.read_csv(r"path/to/groundtruthAnnotations.csv", sep=",")
#print(gt)
# Get the detections on the format [filename, x_min, y_min, x_max, y_max, id_dt]
dt = pd.read_csv(r"path/to/detections.csv", sep=",")

print(dt)

# If needed, upscale detection coordinates (if images were downscaled before inference)
#dt['x_min'] = dt['x_min'] * 8
#dt['y_min'] = dt['y_min'] * 8
#dt['x_max'] = dt['x_max'] * 8
#dt['y_max'] = dt['y_max'] * 8


P = 0 # Positives (meaning total number of objects (e.g. flowers) across all images)
CP = 0 # Correct positives (meaning total number of objects that were correclty detected)
FP = 0 # False positives
FN = 0 # False negatives
MM = 0 # Tracking mismatches



#
##
###
####
# Make sure that the rows/images are in the correct ascending order by ordering by the image number in the filename

gt['frame'] = gt['filename'].str.extract('(\d{6})')
dt['frame'] = dt['filename'].str.extract('(\d{6})')

gt = gt.sort_values('frame')
dt = dt.sort_values('frame')

#print(gt)
#print(dt)

####
###
##
#



#
##
###
####

def calculate_iou(boxA, boxB):
	"""
	Calculates the intersection over union between two bounding boxes. The format of a box is: [x_min, y_min, x_max, y_max]
	"""
	# determine the (x, y)-coordinates of the intersection rectangle
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])

	# compute the area of intersection rectangle
	interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
	if interArea == 0:
		return 0
	# compute the area of both the prediction and ground-truth
	# rectangles
	boxAArea = abs((boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
	boxBArea = abs((boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = interArea / float(boxAArea + boxBArea - interArea)

	# return the intersection over union value
	return iou


def calculate_precision(Correct_positives, False_positives):
	"""
	Calculates the precision of the object detection model
	"""
	Precision = Correct_positives/(Correct_positives + False_positives)
	return Precision


def calculate_recall(Correct_positives, False_negatives):
	"""
	Calculates the recall of the object detection model
	"""
	Recall = Correct_positives/(Correct_positives + False_negatives)
	return Recall


def calculate_f1(Precision, Recall):
	"""
	Calculates the F1 score of the object detection model (based on precision and recall)
	"""
	F1 = (2 * Precision * Recall) / (Precision + Recall)
	return F1


def calculate_mismatch_ratio(Number_of_positives, Number_of_mismatches):
	"""
	Calculates the ratio of mismatches to the numbe of objects
	"""
	MM_Ratio = Number_of_mismatches/Number_of_positives
	return MM_Ratio


def calculate_mota(False_positives, False_negatives, Number_of_mismatches, Number_of_positives):
	"""
	Calculates the mota score of the tracking (based on precision, recall and mismatches)
	"""
	MOTA = 1 - ((False_positives + False_negatives + Number_of_mismatches)/Number_of_positives)
	return MOTA

####
###
##
#


# Create a dataframe of of frames that contain both groundt truths and detections. We'll also create a list of uniqe filenames in the common dataframe
common = gt.merge(dt, on=['filename'], how = 'inner')
# print("Common: \n", common)
filenames_common = common['filename'].unique().tolist() # Get a list of unique filenames to iterate over
#print("Filenames common: ",filenames_common)

# Images containing ground truths but no detections are all counted as false negatives
only_fn = gt[~gt.filename.isin(filenames_common)] # Filenames in gt but not in dt 
FN = FN + only_fn.shape[0] # Add to FN
####

# Images containing detections but no groundt truths are all counted as false positives
only_fp = dt[~dt.filename.isin(filenames_common)] # Filenames in dt but not in gt 
FP = FP + only_fp.shape[0] # Add to FP

##
false_positives_df = only_fp #temp
##

#print("Only fn: \n", only_fn)
#print("Only fp: \n", only_fp)
####


# Create unique ids for every single ground truth bounding box and every detection bounding box
common['id_uni_gt'] = common['filename'] + "_" + common['id_gt']
common['id_uni_tr'] = common['filename'] + "_" + common['id_tr'].astype(str)


common['iou'] = common.apply(lambda x: calculate_iou([x['x_min_x'], x['y_min_x'], x['x_max_x'], x['y_max_x']],[x['x_min_y'], x['y_min_y'], x['x_max_y'], x['y_max_y']]), axis=1)
#print("Common: \n", common)


matches = common[common['iou'] >= iou_threshold]
#print("Matches: \n", matches)


# Get the ground truths that have no matching detections and add them to the false negatives
no_matches = common[common['iou'] < iou_threshold]
false_negatives = no_matches[~no_matches['id_uni_gt'].isin(matches['id_uni_gt'])].drop_duplicates(subset=['id_uni_gt'])
FN = FN + false_negatives.shape[0]

print("False negatives: ", false_negatives)

#print("No matches: ",no_matches)

# Get the detections that have no matching ground truth and add them to the false positives
false_positives = no_matches[~no_matches['id_uni_tr'].isin(matches['id_uni_tr'])].drop_duplicates(subset=['id_uni_tr'])
FP = FP + false_positives.shape[0]

print("False positives: ", false_positives)


##
false_positives_df = false_positives_df.append(false_positives) #temp
##

# Count tracking mismatches

matches_pairs = matches[['id_gt','id_tr']].drop_duplicates()
id_gt_mismatches = matches_pairs.pivot_table(index=['id_gt'], aggfunc='size')-1
id_gt_mismatches_sum = id_gt_mismatches.sum()


print(matches_pairs)
print(id_gt_mismatches)

id_tr_mismatches = matches_pairs.pivot_table(index=['id_tr'], aggfunc='size')-1
id_tr_mismatches_sum = id_tr_mismatches.sum()

print(id_tr_mismatches)

MM = MM + id_gt_mismatches_sum + id_tr_mismatches_sum


#
##
###
#### Count mismatches (MOTA metric)

# Set up the map of the first image
mota_map = pd.DataFrame(columns = ['id_gt', 'id_tr']) # Not used

#print("Reset index")
#print(matches.reset_index(drop = True))

filenames_matches = matches['filename'].unique().tolist()

#print("Filenames matches: \n", filenames_matches)


####
###
##
#



#
##
###
####
P = gt.shape[0]
PR = dt.shape[0]
CP =  PR - FP

print("Number of annotated objects: ", P)
print("Number of predictions made: ", PR )
print("Correct positives: ",CP)
print("Matches nrow: ",matches.shape[0])


precision = calculate_precision(CP,FP)
recall = calculate_recall(CP,FN)
F1 = calculate_f1(precision, recall)
mota = calculate_mota(FP, FN, MM, P)
mm_ratio = calculate_mismatch_ratio(P, MM)

print("False positives: ", FP)
print("False negatives: ", FN)
print("Mismatches: ", MM)

print("Images in the detections: ", len(dt['filename'].unique()))
print("Images in the ground truth: ", len(gt['filename'].unique()))


print("#####", " SCORES ", "#####\n")
print("Precision: ", precision)
print("Recall: ", recall)
print("F1: ", F1)
print("MOTA: ", mota)
print("Mismatch ratio: ", mm_ratio)

####
###
##
#

### END OF SCRIPT ###














