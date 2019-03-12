#!/bin/bash
COMBINED_FILE_NAME=combined.csv
RESULTS_DIR=results
FINAL_FILES_PREFIX=final_info
FINAL_FILE=final_comparison.csv
SUFFIX_TEMP_FILE=temp
TIMES_FILE=tempo.txt
find $RESULTS_DIR -type f -name $COMBINED_FILE_NAME -exec rm {} \;
for dir in `find $RESULTS_DIR -type f -name "$FINAL_FILES_PREFIX*"  | perl -pe "s|/$FINAL_FILES_PREFIX.*$||g" | sort | uniq`
do
	combined_file=$dir/$COMBINED_FILE_NAME
	combined_file_temp=$dir/$COMBINED_FILE_NAME.$SUFFIX_TEMP_FILE
	times_file_temp=$dir/$TIMES_FILE.$SUFFIX_TEMP_FILE
	
	lbase=`echo $dir | perl -pe 's|^\./||g' | perl -pe "s|$FINAL_FILES_PREFIX.*$||g" | perl -pe 's|/|_|g'`
	echo "$lbase"_time_total,"$lbase"_time_running,"$lbase"_time_test >> $times_file_temp
	cat $dir/$TIMES_FILE >> $times_file_temp
	echo "$lbase"_percent,"$lbase"_to_knn >> $combined_file_temp
	find $dir -type f -name "$FINAL_FILES_PREFIX*" -exec grep -inRH --color=auto "\(corretos\|submetidos\)" {} \; | tr -d '|' | awk '{print $NF}' | awk 'NR%2{printf "%s ",$0;next;}1' | tr ' ' ',' >> $combined_file_temp
	paste -d ',' $combined_file_temp $times_file_temp > $combined_file
	rm $combined_file_temp $times_file_temp
done
paste -d ',' `find $RESULTS_DIR -type f -name $COMBINED_FILE_NAME` > $FINAL_FILE
