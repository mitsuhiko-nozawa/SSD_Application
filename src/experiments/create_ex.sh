dnum=$(ls -l | wc -l)
dnum=$(($dnum - 3))
dnum=${dnum//[[:blank:]]}
dsize=${#dnum}

if [ $dsize = 1 ]; then
    dnum=00$dnum
elif [ $dsize = 2 ]; then
    dnum=0$dnum
fi

dname=exp_$dnum
cp -r exp_000 ${dname}