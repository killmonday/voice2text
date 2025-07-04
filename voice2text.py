#coding=utf-8
import logging
import time
import traceback
from typing import List, Literal
import librosa
import numpy as np
from funasr import AutoModel
from tqdm.auto import tqdm
from funasr.utils.postprocess_utils import rich_transcription_postprocess
from flask import Flask, request, Response, jsonify
import requests
from queue import Queue
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
import threading
from tqdm.auto import tqdm
from typing import List, Literal
import io
# with ubuntu22.04:  pip install --ignore-installed -U blinker && pip install flask funasr==0.3.0 requests tqdm soundfile

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
server = Flask(__name__)
task_q = Queue()

callback_url = 'http://192.168.31.13'
bear_token = 'cQTXL8hS9ig5Gc1WFEst4ERUXX3Oi_'

class TranscribeResult:
    """
    Each TranscribeResult object represents one SRT line.
    """

    def __init__(self, text: str, start_time: float, end_time: float):
        self.text = text
        self.start_time = self.float_to_time_format(start_time)
        self.end_time = self.float_to_time_format(end_time)

    def float_to_time_format(self, seconds_float):
        """
        将浮点数转换为视频时间格式（HH:MM:SS）
        
        参数:
            seconds_float (float): 表示秒数的浮点数
            
        返回:
            str: 格式为"HH:MM:SS"的字符串
        """
        # 将浮点数转换为整数秒
        total_seconds = int(round(seconds_float))
        
        # 计算小时、分钟和秒
        hours = total_seconds // 3600
        remaining_seconds = total_seconds % 3600
        minutes = remaining_seconds // 60
        seconds = remaining_seconds % 60
        
        # 格式化为HH:MM:SS，确保每个部分都是两位数
        time_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        
        return time_str

    def __str__(self):
        # return f"text={self.text}, start_time={self.start_time}, end_time={self.end_time})"
        return f"{self.start_time} {self.text}"

    def __repr__(self):
        return str(self)

class Transcriber:
    def __init__(
        self,
        corrector: Literal["opencc", "bert", None] = None,
        offset_in_seconds=-0.25,
        max_length_seconds=5,
        sr=16000,
    ):
        self.corrector = corrector
        self.sr = sr
        self.max_length_seconds = max_length_seconds
        self.offset_in_seconds = offset_in_seconds

    def transcribe(
        self,
        audio_file: str,
    ):
        raise NotImplementedError

 

class AutoTranscriber(Transcriber):
    """
    Transcriber class that uses FunASR's AutoModel for VAD and ASR
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Initialize models
        self.vad_model = AutoModel(
            # model="/root/.cache/modelscope/hub/models/iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
            model="./model_save/speech_fsmn_vad_zh-cn-16k-common-pytorch",
            # model="fsmn-vad",
            max_single_segment_time=30000,
            disable_update=True,
            # device="cpu",
            device="cuda:0",
        )
        self.asr_model = AutoModel(
            # model="/root/.cache/modelscope/hub/models/iic/SenseVoiceSmall",
            model="./model_save/SenseVoiceSmall",
            vad_model=None,
            punc_model=None,
            # ban_emo_unks=True, #不输出表情
            disable_update=True,
            # device="cpu",
            device="cuda:0",
        )

    def transcribe(
        self,
        audio_file: str,
    ) -> List[TranscribeResult]:
        """
        Transcribe audio file to text with timestamps.

        Args:
            audio_file (str): Path to audio file

        Returns:
            List[TranscribeResult]: List of transcription results
        """
        # Load and preprocess audio
        speech, sr = librosa.load(audio_file, sr=self.sr)

        if sr != 16000:
            logger.info('trans to 16000hz..')
            speech = librosa.resample(speech, orig_sr=sr, target_sr=16000)

        # Get VAD segments
        logger.info("Segmenting speech...")

        start_time = time.time()
        vad_results = self.vad_model.generate(input=speech, disable_pbar=True)
        logger.info("VAD took %.2f seconds", time.time() - start_time)

        if not vad_results or not vad_results[0]["value"]:
            return []

        vad_segments = vad_results[0]["value"]

        # Process each segment
        results = []

        start_time = time.time()
        for segment in tqdm(vad_segments, desc="Transcribing"):
            start_sample = int(segment[0] * 16)  # Convert ms to samples
            end_sample = int(segment[1] * 16)
            segment_audio = speech[start_sample:end_sample]

            # Get ASR results for segment
            asr_result = self.asr_model.generate(
                input=segment_audio,
                cache={},
                language="zh",      # "auto" "zn", "en", "yue", "ja", "ko", "nospeech"
                use_itn=True,       # 逆文本规范化，例如把"two thousand twenty three" 处理后变成 "2023"
                batch_size_s=60,    # 音频文件切割为每个60s
            )

            if not asr_result:
                continue

            start_time = max(0, segment[0] / 1000.0 + self.offset_in_seconds)
            end_time = segment[1] / 1000.0 + self.offset_in_seconds

            # Convert ASR result to TranscribeResult format
            segment_result = TranscribeResult(
                text=rich_transcription_postprocess(asr_result[0]["text"]),
                # text=asr_result[0]["text"],
                start_time=start_time,  # Convert ms to seconds
                end_time=end_time,
            )
            results.append(str(segment_result))

        logger.info("ASR took %.2f seconds", time.time() - start_time)

        # Apply Chinese conversion if needed
        start_time = time.time()
        logger.info("Conversion took %.2f seconds", time.time() - start_time)

        return results


transcriber = AutoTranscriber()


@server.route('/', methods=['POST'])
def trans():
    """
    接收音频文件并转换为文本
    """
    file_path = request.form.get('file_path')
    callback_path = request.form.get('callback_path')
    print(file_path, callback_path)
    uid = request.form.get('uid')
    if not uid or not callback_path:
        print(1)
        return jsonify({"error": "uid and callback_path are required"}), 400
    if file_path:
        # 如果提供了文件路径，则直接使用该路径
        file_path = file_path.strip()
    elif 'content' in request.files:
        file = request.files['content']
        print('get file then save..')
        if file.filename == '':
            print(2)
            return jsonify({"error": "No selected file"}), 400

        # 保存文件到临时目录
        file_path = f"./temp/{file.filename}"
        file.save(file_path)
    else:
        print(3)
        return jsonify({"error": "No filepath or file content post"}), 400
    
    task_q.put((file_path, callback_path, uid))
    print(4)
    return jsonify({"status": 'create task ok. wait for result'}), 200


def generate(file_path):
    """
    生成文本
    """
    text = ''
    # 调用模型进行转换
    transcribe_results = transcriber.transcribe(file_path)
    if transcribe_results:
        # for res in transcribe_results:
        #     print(res.start_time, res.text)
        #     print()
        text = "\n".join(transcribe_results)
        return text
    return text

def worker():
    """
    工作线程，处理任务队列中的音频文件
    """
    while True:
        file_path, callback_path, uid = task_q.get()
        bearer_token = bear_token
        headers = {
            'Authorization': f'Bearer {bearer_token}'
        }
        try:
            start_time = time.time()
            text = generate(file_path)
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"Processed {file_path}: {text}\n函数运行时间: {elapsed_time:.4f} 秒")
            url = f"{callback_url}{callback_path}"
            
            files={	'uid':(None,uid),
                'status':(None,'ok'),
                'content': io.StringIO(text)
            }
            
            req = requests.post(url, files=files, headers=headers, timeout=25, verify=False, )
        except:
            print(f'mp3 file {file_path} transcribe error.')
            traceback.print_exc()
            files={	'uid':(None,uid),
                'status':(None,'fail'),
            }
            req = requests.post(url, files=files, headers=headers, timeout=25, verify=False, )
            

if __name__ == "__main__":
    # start_time = time.time()
    # transcriber = AutoTranscriber()
    # transcribe_results = transcriber.transcribe('每天一听-告诉自己一定要全力以赴.mp3')

    # if transcribe_results:
    #     for res in transcribe_results:
    #         print(res.start_time, res.text)
    #         print()
        
    # end_time = time.time()
    # elapsed_time = end_time - start_time
    # print(f"函数运行时间: {elapsed_time:.4f} 秒")
    
    #初始化线程
    for _ in range(1):  # 启动1个工作线程
        threading.Thread(target=worker, daemon=True).start()
    # 启动Flask服务器
    server.run(host='0.0.0.0', port=21111, debug=False, threaded=True, use_reloader=False)
