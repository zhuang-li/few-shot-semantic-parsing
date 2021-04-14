wget -c http://nlp.stanford.edu/data/glove.6B.zip
mkdir -p embedding/glove
unzip glove.6B.zip -d embedding/glove/
rm glove.6B.zip
