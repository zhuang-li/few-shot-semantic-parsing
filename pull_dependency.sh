mkdir -p embedding/glove
wget -c http://nlp.stanford.edu/data/glove.6B.zip
unzip glove.6B.zip -d embedding/glove/
rm glove.6B.zip
