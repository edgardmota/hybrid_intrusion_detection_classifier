#!/bin/bash
COMBINED_FILE_NAME=combined.csv
RESULTS_DIR=results
FINAL_FILES_PREFIX=final_info
FINAL_FILE=final_comparison.csv
SUFFIX_TEMP_FILE=temp
TIMES_FILE=tempo.txt
SEPARATOR='\t'
RNA_ACC_FILE=rna_acc.txt
find $RESULTS_DIR -type f -name $COMBINED_FILE_NAME -exec rm {} \;
for dir in `find $RESULTS_DIR -type f -name "$FINAL_FILES_PREFIX*"  | perl -pe "s|/$FINAL_FILES_PREFIX.*$||g" | sort | uniq`
do
	combined_file=$dir/$COMBINED_FILE_NAME
	combined_file_temp=$dir/$COMBINED_FILE_NAME.$SUFFIX_TEMP_FILE

	lbase=`echo $dir | perl -pe 's|^\./||g' | perl -pe "s|$FINAL_FILES_PREFIX.*$||g" | perl -pe 's|/|_|g' | awk -F'_' '{OFS=FS;$1=$NF=$(NF-1)=$(NF-2)="";print}' | perl -pe 's|(^_+\|_+$)||g'`
	if [ "$1" == "--with-times" ]
	then
		times_file_temp=$dir/$TIMES_FILE.$SUFFIX_TEMP_FILE

		echo -e "$lbase"_time_total"$SEPARATOR$lbase"_time_running"$SEPARATOR$lbase"_time_test >> $times_file_temp
		cat $dir/$TIMES_FILE >> $times_file_temp
	fi
	echo -e "$lbase"_percent"$SEPARATOR$lbase"_to_knn >> $combined_file_temp
	find $dir -type f -name "$FINAL_FILES_PREFIX*" -exec grep -inRH --color=auto "\(corretos\|submetidos\)" {} \; | tr -d '|' | awk '{print $NF}' | awk 'NR%2{printf "%s ",$0;next;}1' | tr ' ' "$SEPARATOR" >> $combined_file_temp
	linhas="$lbase"_sensi"$SEPARATOR$lbase"_especi"\n"
	for file in `find $dir -type f -name "$FINAL_FILES_PREFIX*"`
	do
		sensibilidade=`grep ATAQUE $file  | tail -n1 | perl -pe 's|^.*?(\d+).*?(\d+).*$|\1 \2|g'`
		especificidade=`grep NORMAL $file  | tail -n1 | perl -pe 's|^.*?(\d+).*?(\d+).*$|\1 \2|g'`
		vp=`echo $sensibilidade| awk '{print $1}'`
		fn=`echo $sensibilidade| awk '{print $2}'`
		s=`perl -e "print $vp / ($vp + $fn)"`
		vn=`echo $especificidade| awk '{print $2}'`
		fp=`echo $especificidade| awk '{print $1}'`
		e=`perl -e "print $vn / ($vn + $fp)"`
		linhas="$linhas$s$SEPARATOR$e\n"
	done
	echo -e $linhas | paste $combined_file_temp - > $combined_file
	mv $combined_file $combined_file_temp
	if [ "$1" == "--with-times" ]
	then
		paste -d "$SEPARATOR" $combined_file_temp $times_file_temp > $combined_file
		rm $combined_file_temp $times_file_temp
	else
		mv $combined_file_temp $combined_file
	fi
done
paste -d "$SEPARATOR" `find $RESULTS_DIR -type f -name $COMBINED_FILE_NAME -o -name $RNA_ACC_FILE` > $FINAL_FILE
