# Test time format
#modulename=$lasttesttime,$

if [ "$1" = "all" ]; then
    python3 -m unittest discover
else
    filename=".testtimes" 
    if [ -f "$filename" ]; then
        linnum=$(wc -l $filename  | awk '{print $1}')
        modnum=$(($(find ./tests -name "*.py" | wc -l)-1 ))
        if [ "$linnum" -ne "$modnum" ]; then
            rm $filename
        fi
    fi
    if [ ! -f "$filename" ]; then
        touch $filename
        readarray  -t modulearray  < <(find ./tests/ -name "*.py")
        for i in ${!modulearray[@]}; do 
            modulearray[i]=$(echo ${modulearray[i]} | sed 's/..tests.//g' | sed 's/.py//'); 
        done
        for module in ${modulearray[@]}; do
            if [ "__init__" != $module ]; then
                # echo $module=$(date +%s) >> $ 
                echo "$module=0" >> $filename
            fi
        done
    fi 
    readarray  -t modnametime  < $filename
    for linenum in ${!modnametime[@]}; do
        modulename=$(echo ${modnametime[$linenum]} | sed 's/=.*//')
        moduletesttime=$(echo ${modnametime[$linenum]} | sed 's/.*=//')
        readarray  -t modulefiles  < <(find -name "$modulename.py" | /bin/grep -v tests)
        if [ "${#modulefiles[@]}" -gt "1" ] || [ "${#modulefiles[@]}" -eq "0" ]; then
            :
#            echo "ERROR, several or none modules with same name, number of files = ";
#            echo ${modulename}
        else
            modulemodificationtime=$(stat -c %Y ${modulefiles[0]})
            testfimodificationtime=$(stat -c %Y ./tests/$modulename.py)
#            echo $modulename
#            echo $moduletesttime
#            echo $modulemodificationtime
#            echo $testfimodificationtime
#            echo "------------------------"
            if  [ "$modulemodificationtime" -gt "$moduletesttime" ] ||\
                [ "$testfimodificationtime" -gt "$moduletesttime" ];  then
                python3 -m unittest -v -f tests.${modulename}
                modnametime[$linenum]="${modulename}=$(date +%s)"
            fi
        fi

    done
    printf "%s\n" "${modnametime[@]}" > $filename

fi
