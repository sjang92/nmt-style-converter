# Writing Style Converter

## Mark Kwon, Se Won Jang, Jesse Min
## CS224N, Stanford University


Our writing style converter is a deep-learning based, TensorFlow powered, writing style converter.

  - TensorFlow
  - Machine Translation based on RNN
  - More to come! Stay tuned!

#### Building for source
For data gathering:
```sh
$ cd dataCollecctor
$ sh gatherData.sh (gmail_id) (gmail_pw) (path_to_file_to_read)
```
For example, after changing the directory to dataCollector:
```sh
$ sh gatherData.sh janeDoe abcd1234! sampleTest/simple_single.txt
```
