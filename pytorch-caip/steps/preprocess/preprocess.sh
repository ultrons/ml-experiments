
function usage() {
        echo "Usage: ./preprocess.sh "
        echo "--dataset_bucket=gs://<WHERE PREPROCESSED DATA IS WRITTEN> "
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
      --dataset_bucket)DATASET_BUCKET=$2
                       shift
                       ;;
      --args_in)  shift;
                  ARGS_IN=$@
                       ;;
      --args_out)  ARGS_OUT=$2
                   shift
                       ;;
       *)              shift
                        ;;
    esac
done   

# Check if the dataset_bucket input was provided
if [ "${DATASET_BUCKET}" = "" ]; then
        echo "ERROR: No dataset_bucket specified!!!"
        usage
        exit 1
# Check if the Bucket exists/Accessible
else 
        BUCKET_NAME=`echo ${DATASET_BUCKET} | sed 's@\(gs://[^/]*\).*@\1@'`
        gsutil -q ls ${BUCKET_NAME} 1>/dev/null ||  invalid_bucket ${BUCKET_NAME}
fi

# Pre-Processing Code

#cd /fairseq/examples/language_model/  && bash prepare-wikitext-103.sh
cd /fairseq
#TEXT=examples/language_model/wikitext-103 && fairseq-preprocess \
#    --only-source \
#    --trainpref $TEXT/wiki.train.tokens \
#    --validpref $TEXT/wiki.valid.tokens \
#    --testpref $TEXT/wiki.test.tokens \
#    --destdir data-bin/wikitext-103 \
#    --workers 20

#gsutil -m cp data-bin ${DATASET_BUCKET}/ || exit 1 

mkdir -p `dirname $ARGS_OUT` 
DATSET_ARGS="[\"--dataset\", \"$DATASET_BUCKET/data-bin\",  \"--\", \"data-bin/wikitext-103\"]"
python -c "import ast; dataset_args =$DATSET_ARGS; dataset_args.extend(ast.literal_eval('$ARGS_IN')); print(dataset_args)" >  $ARGS_OUT
#python -c "import ast; dataset_args =$DATSET_ARGS; print(dataset_args)" >  $ARGS_OUT
#python -c "import ast; dataset_args =$DATSET_ARGS; dataset_args.append('a'); print(dataset_args)" >  $ARGS_OUT
