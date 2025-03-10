#!/bin/bash

image_data=$(base64 -w 0 /home/stardust/test-images/400.png)
echo '{"image": "'"$image_data"'"}' > data.json
# echo $image_data > data.json
curl -X POST -d @data.json http://0.0.0.0:33847/