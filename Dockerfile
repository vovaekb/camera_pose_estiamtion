FROM python:3
ADD pose_estimation_class.py
ADD affnet_descriptor.py
RUN wget https://github.com/ducha-aiki/affnet/raw/master/convertJIT/AffNetJIT.pt \
    && wget https://github.com/ducha-aiki/affnet/raw/master/convertJIT/OriNetJIT.pt \
    && wget https://github.com/ducha-aiki/affnet/raw/master/test-graf/H1to6p
RUN pip install -r requirements.txt
