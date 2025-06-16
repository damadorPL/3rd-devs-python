import os
import json
import subprocess
import asyncio
import re
from typing import List, Dict, Any, Optional, Union, TypedDict
import ffmpeg

class AudioMetadata(TypedDict):
    duration: float
    sample_rate: int
    channels: int
    bit_rate: int
    codec: str
    format: str

class AudioLoudnessData(TypedDict):
    time: float
    loudness: float

class SilenceInterval(TypedDict):
    start: float
    end: float
    duration: float

class AudioChunk(TypedDict):
    start: float
    end: float

class NonSilentInterval(TypedDict):
    start: float
    end: float
    duration: float

class AudioService:
    async def get_metadata(self, file_path: str) -> AudioMetadata:
        try:
            data = await self.probe_file(file_path)
            return self.extract_metadata(data)
        except Exception as error:
            self.handle_error(error)

    async def probe_file(self, file_path: str) -> Dict[str, Any]:
        try:
            probe = await asyncio.create_subprocess_exec(
                'ffprobe', '-v', 'error', '-of', 'json', '-show_format', '-show_streams', file_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await probe.communicate()
            if probe.returncode != 0:
                raise Exception(f'FFprobe error: {stderr.decode()}')
            return json.loads(stdout.decode())
        except Exception as error:
            raise Exception(f'Error probing file: {str(error)}')

    def extract_metadata(self, data: Dict[str, Any]) -> AudioMetadata:
        stream = next((s for s in data.get('streams', []) if s.get('codec_type') == 'audio'), None)
        if not stream:
            raise Exception('No audio stream found')
        format_info = data.get('format', {})
        return {
            'duration': float(format_info.get('duration', 0)),
            'sample_rate': int(stream.get('sample_rate', 0)),
            'channels': int(stream.get('channels', 0)),
            'bit_rate': int(stream.get('bit_rate', 0)),
            'codec': stream.get('codec_name', 'unknown'),
            'format': format_info.get('format_name', 'unknown')
        }

    def handle_error(self, error: Exception) -> None:
        print(f'Error getting audio metadata: {str(error)}')
        raise error

    async def analyze_loudness(self, file_path: str, interval: float = 0.1) -> List[AudioLoudnessData]:
        loudness_data = []
        try:
            process = await asyncio.create_subprocess_exec(
                'ffmpeg', '-i', file_path, '-af', f'astats=metadata=1:reset={interval},aresample=8000', '-f', 'null', '-',
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            _, stderr = await process.communicate()
            for line in stderr.decode().splitlines():
                rms_match = re.search(r'lavfi\.astats\.Overall\.RMS_level=(-?\d+(\.\d+)?)', line)
                time_match = re.search(r'pts_time:(\d+(\.\d+)?)', line)
                if rms_match and time_match:
                    loudness_data.append({
                        'time': float(time_match.group(1)),
                        'loudness': float(rms_match.group(1))
                    })
            return loudness_data
        except Exception as error:
            raise Exception(f'Error analyzing loudness: {str(error)}')

    async def detect_silence(self, file_path: str, threshold: int = -50, min_duration: int = 2) -> List[SilenceInterval]:
        silence_intervals = []
        current_interval = {}
        try:
            process = await asyncio.create_subprocess_exec(
                'ffmpeg', '-i', file_path, '-af', f'silencedetect=noise={threshold}dB:d={min_duration}', '-f', 'null', '-',
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            _, stderr = await process.communicate()
            for line in stderr.decode().splitlines():
                silence_start_match = re.search(r'silence_start: ([\d\.]+)', line)
                silence_end_match = re.search(r'silence_end: ([\d\.]+) \| silence_duration: ([\d\.]+)', line)
                if silence_start_match:
                    current_interval['start'] = float(silence_start_match.group(1))
                elif silence_end_match:
                    current_interval['end'] = float(silence_end_match.group(1))
                    current_interval['duration'] = float(silence_end_match.group(2))
                    silence_intervals.append(current_interval.copy())
                    current_interval = {}
            return silence_intervals
        except Exception as error:
            raise Exception(f'Error detecting silence: {str(error)}')

    async def detect_non_silence(self, file_path: str, threshold: int = -50, min_duration: int = 2) -> List[NonSilentInterval]:
        silence_intervals = []
        non_silent_intervals = []
        total_duration = None
        try:
            process = await asyncio.create_subprocess_exec(
                'ffmpeg', '-i', file_path, '-af', f'silencedetect=noise={threshold}dB:d={min_duration}', '-f', 'null', '-',
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            _, stderr = await process.communicate()
            for line in stderr.decode().splitlines():
                silence_start_match = re.search(r'silence_start: ([\d\.]+)', line)
                silence_end_match = re.search(r'silence_end: ([\d\.]+) \| silence_duration: ([\d\.]+)', line)
                duration_match = re.search(r'Duration: (\d{2}):(\d{2}):(\d{2}\.\d{2})', line)
                if silence_start_match:
                    silence_intervals.append({'start': float(silence_start_match.group(1)), 'end': 0, 'duration': 0})
                elif silence_end_match:
                    last_interval = silence_intervals[-1]
                    last_interval['end'] = float(silence_end_match.group(1))
                    last_interval['duration'] = float(silence_end_match.group(2))
                elif duration_match:
                    hours, minutes, seconds = map(float, duration_match.groups())
                    total_duration = hours * 3600 + minutes * 60 + seconds
            if total_duration is None:
                raise Exception('Could not determine audio duration')
            last_end = 0
            for silence in silence_intervals:
                if silence['start'] > last_end:
                    non_silent_intervals.append({
                        'start': last_end,
                        'end': silence['start'],
                        'duration': silence['start'] - last_end
                    })
                last_end = silence['end']
            if last_end < total_duration:
                non_silent_intervals.append({
                    'start': last_end,
                    'end': total_duration,
                    'duration': total_duration - last_end
                })
            return non_silent_intervals
        except Exception as error:
            raise Exception(f'Error detecting non-silence: {str(error)}')

    async def get_average_silence_threshold(self, file_path: str) -> float:
        try:
            process = await asyncio.create_subprocess_exec(
                'ffprobe', '-v', 'error', '-of', 'json', '-show_format', '-show_streams', file_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            if process.returncode != 0:
                raise Exception(f'FFprobe error: {stderr.decode()}')
            data = json.loads(stdout.decode())
            audio_stream = next((s for s in data.get('streams', []) if s.get('codec_type') == 'audio'), None)
            if not audio_stream:
                raise Exception('No audio stream found')
            rms_level = float(audio_stream.get('rms_level', -60))
            silence_threshold = rms_level + 10
            return silence_threshold
        except Exception as error:
            print(f'Error: {str(error)}')
            raise error

    async def get_average_silence_duration(self, file_path: str) -> float:
        average_silence_threshold = await self.get_average_silence_threshold(file_path)
        silence_segments = await self.detect_silence(file_path, average_silence_threshold + 25, 1)
        if not silence_segments:
            return 0
        total_silence_duration = sum(segment['end'] - segment['start'] for segment in silence_segments)
        return total_silence_duration / len(silence_segments)

    def extract_non_silent_chunks(self, silence_segments: List[SilenceInterval], total_duration: float) -> List[AudioChunk]:
        non_silent_chunks = []
        last_end = 0
        for silence in silence_segments:
            if silence['start'] > last_end:
                non_silent_chunks.append({'start': last_end, 'end': silence['start']})
            last_end = silence['end']
        if last_end < total_duration:
            non_silent_chunks.append({'start': last_end, 'end': total_duration})
        return non_silent_chunks

    async def save_non_silent_chunks(self, file_path: str, chunks: List[AudioChunk]) -> List[str]:
        output_dir = os.path.join(os.path.dirname(__file__), 'storage', 'chunks')
        os.makedirs(output_dir, exist_ok=True)
        output_paths = []
        for i, chunk in enumerate(chunks):
            output_path = os.path.join(output_dir, f'chunk_{i}.wav')
            try:
                process = await asyncio.create_subprocess_exec(
                    'ffmpeg', '-i', file_path, '-ss', str(chunk['start']), '-t', str(chunk['end'] - chunk['start']), output_path,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                await process.communicate()
                if process.returncode == 0:
                    output_paths.append(output_path)
            except Exception as error:
                print(f'Error saving chunk {i}: {str(error)}')
        return output_paths

    async def process_and_save_non_silent_chunks(self, file_path: str) -> List[str]:
        metadata = await self.get_metadata(file_path)
        silence_intervals = await self.detect_silence(file_path)
        non_silent_chunks = self.extract_non_silent_chunks(silence_intervals, metadata['duration'])
        return await self.save_non_silent_chunks(file_path, non_silent_chunks)

    async def convert_wav_to_ogg(self, input_path: str, output_path: str) -> None:
        try:
            process = await asyncio.create_subprocess_exec(
                'ffmpeg', '-i', input_path, '-c:a', 'libvorbis', output_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            await process.communicate()
            if process.returncode != 0:
                raise Exception('Error converting WAV to OGG')
        except Exception as error:
            raise Exception(f'Error converting WAV to OGG: {str(error)}')

    async def split(self, file_path: str, silence_threshold_offset: int = 25) -> List[str]:
        min_silence_duration = (await self.get_average_silence_duration(file_path)) * 0.9
        average_silence_threshold = await self.get_average_silence_threshold(file_path)
        non_silent_chunks = await self.detect_non_silence(file_path, int(average_silence_threshold + silence_threshold_offset), min_silence_duration)
        non_silent_chunks = [chunk for chunk in non_silent_chunks if chunk['duration'] >= 1]

        chunks = await self.save_non_silent_chunks(file_path, non_silent_chunks)
        ogg_chunks = []

        for chunk in chunks:
            ogg_chunk = chunk.replace('.wav', '.ogg')
            if not chunk.lower().endswith('.ogg'):
                await self.convert_to_ogg(chunk, ogg_chunk)
                os.remove(chunk)
            else:
                os.copyfile(chunk, ogg_chunk)

            stats = os.stat(ogg_chunk)
            if stats.st_size > 20 * 1024 * 1024:  # 20MB limit
                os.remove(ogg_chunk)
                raise Exception(f'File {ogg_chunk} is too big ({stats.st_size} bytes)')

            ogg_chunks.append(ogg_chunk)

        return ogg_chunks

    async def convert_to_ogg(self, input_path: str, output_path: str) -> None:
        try:
            process = await asyncio.create_subprocess_exec(
                'ffmpeg', '-i', input_path, '-c:a', 'libvorbis', output_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            await process.communicate()
            if process.returncode != 0:
                raise Exception('Error converting to OGG')
        except Exception as error:
            raise Exception(f'Error converting to OGG: {str(error)}')
