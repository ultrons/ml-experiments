FROM gcr.io/deeplearning-platform-release/pytorch-gpu.1-4

RUN git clone http://github.com/ultrons/fairseq -b fairseq-caip

WORKDIR /fairseq

RUN pip install --editable .
RUN pip install fastBPE sacremoses

COPY ./preprocess.sh /


ENTRYPOINT ["bash",  "/preprocess.sh" ]
