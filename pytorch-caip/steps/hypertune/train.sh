usage(){
        echo "Usage: ./train.sh "
        echo "--dataset=gs://<full-path-to-preprocessed-data> \\"
	echo "-- \\"
	echo "<options for fairseq-train>"
}

while [ $# -ne 0 ]; do
    case "$1" in
	    -h|--help) usage
                       exit
		       shift
                       ;;
      --dataset)dataset=$2
                       shift
                       ;;
                   --) shift
                       break
                       ;;
       *)              shift
                        ;;
    esac
done   


function invalid_bucket () {
	echo "ERROR: Invalid/Inaccessible Bucket: ${1}" 
      	usage 
       	exit 1
}

function invalid_dataset () {
	echo "ERROR: Invalid/Inaccessible Dataset: ${1}" 
      	usage 
       	exit 1
}

# Check if the dataset-bucket input was provided
if [ "${dataset}" = "" ]; then
	echo "ERROR: No dataset-bucket specified!!!"
	usage
	exit 1
# Check if the Bucket exists/Accessible
else 
	BUCKET_NAME=`echo ${dataset} | sed 's@\(gs://[^/]*\).*@\1@'`
	gsutil -q ls ${BUCKET_NAME} 1>/dev/null ||  invalid_bucket ${BUCKET_NAME}
	gsutil -m cp -r ${dataset} . || invalid_dataset
fi

dataset_local=`basename ${dataset}`


echo "faiseq-train ${dataset_local} $@"

