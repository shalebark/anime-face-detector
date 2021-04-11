# download the model

# snippet from https://gist.github.com/darencard/079246e43e3c4b97e373873c6c9a3798
# author: darencard
# url: https://gist.github.com/darencard/079246e43e3c4b97e373873c6c9a3798
gURL='https://drive.google.com/open?id=1WjBgfOUqp4sdRd9BHs4TkdH2EcBtV5ri'
# match more than 26 word characters
ggID=$(echo "$gURL" | egrep -o '(\w|-){26,}')

ggURL='https://drive.google.com/uc?export=download'

curl -sc /tmp/gcokie "${ggURL}&id=${ggID}" >/dev/null
getcode="$(awk '/_warning_/ {print $NF}' /tmp/gcokie)"

curl --insecure -C - -LJb /tmp/gcokie "${ggURL}&confirm=${getcode}&id=${ggID}" > model/res101_faster_rcnn_iter_60000.zip

# unzip the file to model
unzip model/res101_faster_rcnn_iter_60000.zip -d model/

# build
make clean
make

# removing the ziped model
rm -f model/res101_faster_rcnn_iter_60000.zip
