package com.example.sound.devicesound;

import android.media.AudioFormat;
import android.media.AudioRecord;
import android.media.MediaRecorder;
import android.util.Log;

import org.apache.commons.math3.complex.Complex;
import org.apache.commons.math3.transform.*;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static java.lang.Math.abs;
import static java.lang.Math.min;

public class Listentone {

    int HANDSHAKE_START_HZ = 4096;
    int HANDSHAKE_END_HZ = 5120 + 1024;

    int START_HZ = 1024;
    int STEP_HZ = 256;
    int BITS = 4;

    int FEC_BYTES = 4;

    private int mAudioSource = MediaRecorder.AudioSource.MIC;
    private int mSampleRate = 44100;
    private int mChannelCount = AudioFormat.CHANNEL_IN_MONO;
    private int mAudioFormat = AudioFormat.ENCODING_PCM_16BIT;
    private float interval = 0.1f;

    private int mBufferSize = AudioRecord.getMinBufferSize(mSampleRate, mChannelCount, mAudioFormat);

    public AudioRecord mAudioRecord = null;
    int audioEncodig;
    boolean startFlag;
    FastFourierTransformer transform;

    // def dominanat(frame_rate, chunk):
    private double findFrequency(double[] toTransform) {
        int len = toTransform.length;
        double[] real = new double[len];
        double[] img = new double[len];
        double realNum; // 실수부
        double imgNum; // 허수부
        double[] mag = new double[len];


        Complex[] complx = transform.transform(toTransform, TransformType.FORWARD);
        // #freqs = np.fft.fftfreq(len(chunk))
        Double[] freq = this.fftfreq(complx.length, 1);

        for (int i = 0; i < complx.length; i++) {
            realNum = complx[i].getReal();
            imgNum = complx[i].getImaginary();
            // #w = np.fft.fft(chunk)
            // 실수부와 허수부로 이루어진 fft 변환된 signal array를 얻을 수 있도록 함
            mag[i] = Math.sqrt((realNum * realNum) + (imgNum * imgNum));
        }
        // #peak_coeff = np.argmax(np.abs(w))
        double peak_coeff = 0;
        int argmax = 0;
        // argmax() 함수는 np.abs(w) 즉 w의 절대값 중 가장 큰 값의 인덱스를 구하는 함수인데 JAVA로는 다음과 같이 인덱스를 얻는다.
        for(int i = 0 ; i< complx.length; i++) {
            if(peak_coeff < mag[i]) {
                argmax = i;
                peak_coeff = mag[argmax];
            }
        }
        Double peak_freq = freq[argmax];
        // #return abs(peak_freq * frame_rate) # in Hz
        // import static java.lang.Math.abs; 으로 절대값 구할 수 있도록 함
        // mSampleRate의 경우 decode.py의 frame_rate에 해당 | 44100
        return abs(peak_freq * mSampleRate);
    }

    // numpy.fft.fftfreq()
    private Double[] fftfreq(int length, int sample_spacing) {
        double val = 1.0 / (length * sample_spacing); // #val = 1.0 / (n * d)
        // #results = empty(n, int)
        int [] results = new int[length];
        // #N = (n-1)//2 + 1
        int N = (length - 1) / 2 + 1;
        // #results[:N]= arange(0, N, dtype=int)
        for(int i = 0; i <= N; i++) {
            results[i] = i;
        }
        int NN = - (length / 2);
        // #results[N:] = arange(-(n//2), 0, dtype=int)
        for(int i = N + 1; i < length ; i++) {
            results[i] = NN;
            NN++;
        }
        Double [] results_return = new Double[length]; // Double은 객체타입이며 new를 통해 생성해서 Double 객체가 가지고 있는 메서드 및 멤버들을 사용
        // #results * val
        for(int i = 0; i < length; i++) {
            results_return[i] = results[i] * val;
        }
        return results_return;
    }


    public Listentone(){

        transform = new FastFourierTransformer(DftNormalization.STANDARD);
        startFlag = false;
        mAudioRecord = new AudioRecord(mAudioSource, mSampleRate, mChannelCount, mAudioFormat, mBufferSize);
        mAudioRecord.startRecording();


    }

    public int findPowerSize(int blocksize_in) { // 4096을 반환 - 2205와 가장 근접한 2의 제곱수는 4096이 맞긴함
        int power = 1;
        do {
            power *= 2;
        } while(power < blocksize_in);
        return power;
    }


    // #def match(freq1, freq2):
    public boolean match(double freq1, double freq2) {
    // #return abs(freq1 - freq2) < 20
        return abs(freq1 - freq2) < 20;
    }

    //def extract_packet(freqs):
    public List<Integer> extract_packet(Double [] freqs_in) {
        int freq_length = freqs_in.length;
        double [] freqs = new double[freq_length];
        int [] bit_chunks = new int[freq_length];

        int half = 1;
        // #freqs = freqs[::2]
        for(int i = 0; i < freq_length / 2; i++) {
            freqs[i] = freqs_in[half];
            half = half + 2;
        }
        // #bit_chunks = [int(round((f - START_HZ) / STEP_HZ)) for f in freqs]
        for(int i = 0; i < freq_length / 2; i++) {
            bit_chunks[i] = (int) Math.round((freqs[i] - START_HZ) / STEP_HZ);
        }
        int [] bit_chunks_calc = new int[freq_length];
        int index = 0;
        for(int i = 1; i < freq_length / 2; i++) {
            if(bit_chunks[i] >= 0 && bit_chunks[i] <= Math.pow(2, BITS)) {
                bit_chunks_calc[index] = bit_chunks[i];
                index++;
            }
        }

        // #return bytearray(decode_bitchunks(BITS, bit_chunks))
        List<Integer> chunks_fin = decode_bitchunks(BITS, bit_chunks_calc);
        return chunks_fin;
    }

    // def decode_bitchunks(chunk_bits, chunks):
    public List<Integer> decode_bitchunks(int chunk_bits, int [] chunks) {
        // #out_bytes = []
        List<Integer> out_bytes = new ArrayList<>();
        // #next_read_chunk = 0
        int next_read_chunk = 0;
        // #next_read_bit = 0
        int next_read_bit = 0;

        // #byte = 0
        int to_byte = 0;
        // #bits_left = 8
        int bits_left = 8;


        int chunks_length = chunks.length;
        // #while next_read_chunk < len(chunks):
        while(next_read_chunk < chunks_length) {
            // #can_fill = chunk_bits - next_read_bit
            int can_fill = chunk_bits - next_read_bit;
            // #to_fill = min(bits_left, can_fill)
            int to_fill = min(bits_left, can_fill);
            // #offset = chunk_bits - next_read_bit - to_fill
            int offset = chunk_bits - next_read_bit - to_fill;
            // #byte <<= to_fill
            to_byte <<= to_fill;
            // #shifted = chunks[next_read_chunk] & (((1 << to_fill) - 1) << offset)
            int shifted = chunks[next_read_chunk] & (((1 << to_fill) - 1) << offset);
            // #byte |= shifted >> offset;
            to_byte |= shifted >> offset;
            // #bits_left -= to_fill
            bits_left -= to_fill;
            // #next_read_bit += to_fill
            next_read_bit += to_fill;

            // #if bits_left <= 0:
            if (bits_left <= 0) {
                // #out_bytes.append(byte)
                out_bytes.add(to_byte);
                // #byte = 0
                to_byte = 0;
                // #bits_left = 8
                bits_left = 8;
            }

            // #if next_read_bit >= chunk_bits:
            if (next_read_bit >= chunk_bits){
                // #next_read_chunk += 1
                next_read_chunk += 1;
                // #next_read_bit -= chunk_bits
                next_read_bit -= chunk_bits;
            }
        }
        // #return out_bytes
        return out_bytes;
    }

// def listen_linux(frame_rate=44100, interval=0.1):
public void PreRequest() {
    // findPowerSize 함수를 쓰는 이유는 buffersize에 2의 제곱수 형태로 들어가야 하기 때문
    int blocksize = findPowerSize((int) (long) Math.round(interval / 2 * mSampleRate)); // #listen_linux의 num_frames에 해당


    //Log.d("blocksize", ""+blocksize); // 4096이 뜨는데 2205와 가장 가까운 2의 제곱수는 4096이 맞음


    short[] buffer = new short[blocksize]; // buffer에 소리데이터가 들어감

    // #packet = []
    List<Double> packet = new ArrayList<>();
    List<Integer> byte_stream = new ArrayList<>();

        while(true) {
            // #l, data = mic.read()
            int bufferedReadResult = mAudioRecord.read(buffer, 0, blocksize);
            // #if not l:
            if (bufferedReadResult < 0) { // 양의정수만
                // #continue
                continue;
            }
            double[] chunk = new double[blocksize]; // #chunk = np.fromstring(data, dtype=np.int16)
            // #chunk = np.fromstring(data, dtype=np.int16)
            for (int i = 0; i < blocksize; i++) {
                chunk[i] = buffer[i];
            }

            // #dom = dominant(frame_rate, chunk)
            double dom = findFrequency(chunk);
            Log.d("dom : ", Double.toString(dom));

            // #if in_packet and match(dom, HANDSHAKE_END_HZ):
            if (startFlag && match(dom, HANDSHAKE_END_HZ)) {

                /* decode.py에서 reed solomon으로 encode된 데이터들을 decode
                   에러가 있다면 해당 원소를 출력
                   이게 없어서 깨지게 출력될 수 있음 // 상관없
                #try:
                #    byte_stream = RSCodec(FEC_BYTES).decode(byte_stream)
                #    byte_stream = byte_stream.decode("utf-8")
                #    display(byte_stream)
                #    display("")
                #except ReedSolomonError as e:
                #    print("{}: {}".format(e, byte_stream))
                 */
                Log.d("packet : ", packet.toString());
                Double[] packet_to_array = packet.toArray(new Double[packet.size()]);
                // #byte_stream = extract_packet(packet)
                byte_stream = extract_packet(packet_to_array);
                Log.d("List data : ", Arrays.toString(packet_to_array));
                String sentence = "";
                for (int i = 0; i < byte_stream.size(); i++) {
                    sentence = sentence + Character.toString((char) ((int) byte_stream.get(i)));
                }
                Log.d("test", sentence);
                // #packet = []
                packet.clear();
                byte_stream.clear();
                // #in_packet = False
                startFlag = false;
                // #elif in_packet:
            } else if (startFlag) {
                // #packet.append(dom)
                packet.add(dom);
                // #elif match(dom, HANDSHAKE_START_HZ):
            } else if (match(dom, HANDSHAKE_START_HZ)) {
                // #in_packet = True
                startFlag = true;
            }

        }
    }
}
