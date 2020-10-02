#!/bin/sh -x

function usage() {
        echo "Usage: ./preprocess.sh "
        echo "--dataset-bucket=gs://<WHERE PREPROCESSED DATA IS WRITTEN> "
}

function invalid_bucket () {
	echo "ERROR: Invalid/Inaccessible Bucket ${1}" 
      	usage 
       	exit 1
}

while [ $# -ne 0 ]; do
    case "$1" in
	    -h|--help) usage
                       exit
		       shift
                       ;;
      --dataset-bucket)DATASET_BUCKET=$2
                       shift
                       ;;
       *)              shift
                        ;;
    esac
done   

# Check if the dataset-bucket input was provided
if [ "${DATASET_BUCKET}" = "" ]; then
	echo "ERROR: No dataset-bucket specified!!!"
	usage
	exit 1
# Check if the Bucket exists/Accessible
else 
	BUCKET_NAME=`echo ${DATASET_BUCKET} | sed 's@\(gs://[^/]*\).*@\1@'`
	gsutil -q ls ${BUCKET_NAME} 1>/dev/null ||  invalid_bucket ${BUCKET_NAME}
fi

# Pre-Processing Code

cd /fairseq/examples/language_model/  && bash prepare-wikitext-103.sh
cd /fairseq
TEXT=examples/language_model/wikitext-103 && fairseq-preprocess \
    --only-source \
    --trainpref $TEXT/wiki.train.tokens \
    --validpref $TEXT/wiki.valid.tokens \
    --testpref $TEXT/wiki.test.tokens \
    --destdir data-bin/wikitext-103 \
    --workers 20

gsutil -m cp data-bin ${DATASET_BUCKET}/ || exit 1 
