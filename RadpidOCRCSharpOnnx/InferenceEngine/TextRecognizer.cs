using Microsoft.ML.OnnxRuntime;
using OpenCvSharp;
using RadpidOCRCSharpOnnx.Config;
using System;
using System.Collections.Generic;
using System.Text;
using static System.Runtime.InteropServices.JavaScript.JSType;

namespace RadpidOCRCSharpOnnx.InferenceEngine
{
    public class TextRecognizer
    {
        private OrtInferSession _session;
        private readonly float[] _batchData;
        private readonly float[] imgData;
        private readonly List<string> _charList;
        public TextRecognizer()
        {
            _batchData = new float[RecConfig.RecBatchNum * RecConfig.RecImgShape[0] * RecConfig.RecImgShape[1] * RecConfig.RecImgShape[2]];
            imgData = new float[RecConfig.RecImgShape[0] * RecConfig.RecImgShape[1] * RecConfig.RecImgShape[2]];
            _session = new OrtInferSession(RecConfig.ModelPath);
            if (_session.HaveKey())
            {
                _charList = _session.GetCharacterList();
            }
            if (_charList == null || _charList.Count == 0)
            {
                _charList = ReadCharacterFile();
            }

        }

        public void RecognizeTxt(Mat[] imgList)
        {
            int[] indices = new int[imgList.Length];
            float[] widthList = new float[imgList.Length];
            for (int i = 0; i < indices.Length; i++)
            {
                indices[i] = i;
                widthList[i] = (float)imgList[i].Width / (float)imgList[i].Height;
            }

            Array.Sort(indices, (a, b) => widthList[a].CompareTo(widthList[b]));
            int imgCount = imgList.Length;

            InferenceResult[] rec_res = new InferenceResult[imgCount];
            for (int i = 0; i < imgCount; i++)
            {
                rec_res[i] = new InferenceResult("", 0.0f);
            }
            int img_c = RecConfig.RecImgShape[0];
            int img_h = RecConfig.RecImgShape[1];
            int img_w = RecConfig.RecImgShape[2];

            int idx = 0;

            for (int i = 0; i < imgCount; i += RecConfig.RecBatchNum)
            {
                int endNo = Math.Min(imgCount, i + RecConfig.RecBatchNum);
                int batchSize = endNo - i;
                float[] batchData = _batchData;

                if (batchSize != RecConfig.RecBatchNum)
                {
                    batchData = new float[batchSize * img_c * img_h * img_w];
                }
                float[] wh_ratio_list = new float[batchSize];
                float max_wh_ratio = (float)img_w / (float)img_h;

                for (int j = i, ratioIdx = 0; j < endNo; j++, ratioIdx++)
                {
                    float wh_ratio = (float)imgList[indices[j]].Width / (float)imgList[indices[j]].Height;
                    max_wh_ratio = Math.Max(max_wh_ratio, wh_ratio);
                    wh_ratio_list[ratioIdx] = wh_ratio;
                }

                idx = 0;
                for (int j = i; j < endNo; j++)
                {
                    idx = ResizeNormImg(imgList[indices[j]], idx, batchData, max_wh_ratio);
                }

                var input = new DataTensorDimensions(batchData, new long[] { batchSize, 3, img_h, img_w });
                using var outData = _session.RunInference(input);
            }
        }


        public int ResizeNormImg(Mat img, int idx, float[] inputData, float max_wh_ratio)
        {
            // 获取原图尺寸和通道数
            int h = img.Height;
            int w = img.Width;
            int channels = img.Channels();
            int img_c = RecConfig.RecImgShape[0];
            int img_h = RecConfig.RecImgShape[1];
            int img_w = RecConfig.RecImgShape[2];

            if (img_c != channels)
                throw new ArgumentException($"The count of image channels does not match：expect {img_c}，actual {channels}");

            img_w = (int)(img_h * max_wh_ratio);

            // 计算缩放后的宽度（保持宽高比，但不超过目标宽度）
            float ratio = (float)w / h;
            double estimatedWidth = Math.Ceiling(img_h * ratio);

            int resized_w = estimatedWidth > img_w ? img_w : (int)estimatedWidth;

            // 缩放图像到 (resized_w, img_h)
            using Mat resized = new Mat();
            Cv2.Resize(img, resized, new OpenCvSharp.Size(resized_w, img_h));

            Array.Fill(imgData, 0.0f);

            for (int i = 0; i < img_c; i++)
            {
                for (int j = 0; j < img_h; j++)
                {
                    for (int k = 0; k < img_w; k++)
                    {
                        if (k < resized_w)
                        {
                            var val = (float)resized.At<Vec3b>(j, k)[i];
                            val = (val / 255.0f) * 2f - 1f;
                            inputData[idx++] = val;
                        }
                        else
                        {
                            inputData[idx++] = 0.0f;
                        }
                    }
                }
            }
            return idx;
        }

        public void RecPostProcess(OrtValue ortValue)
        {
            var shapeInfo = ortValue.GetTensorTypeAndShape();
            int batchSize = (int)shapeInfo.Shape[0];
            int tNum = (int)shapeInfo.Shape[1];
            int numClasses = (int)shapeInfo.Shape[2];

            var data = ortValue.GetTensorDataAsSpan<float>();

            var maxIndexAndValue = GetMaxIndexAndValue(data, batchSize, tNum, numClasses);
            int[] ignored_tokens = getIgnoredTokens();
            for (int i = 0; i < batchSize; i++)
            {
                int[] token_indices = maxIndexAndValue.Indices[i];
                bool[] selection = new bool[token_indices.Length];
                Array.Fill(selection, true);
            }
        }

        public ValueTuplePairArray GetMaxIndexAndValue(ReadOnlySpan<float> data, int batchSize, int tNum, int numClasses)
        {
            float[][] maxValues = new float[batchSize][];
            int[][] maxIndices = new int[batchSize][];
            int maxIdx = 0;
            float maxVal = float.MinValue;
            int idx = 0;
            for (int i = 0; i < batchSize; i++)
            {
                maxValues[i] = new float[tNum];
                maxIndices[i] = new int[tNum];

                for (int j = 0; j < tNum; j++)
                {
                    maxIdx = 0;
                    maxVal = float.MinValue;

                    for (int k = 0; k < numClasses; k++)
                    {
                        float val = data[idx++];
                        if (val > maxVal)
                        {
                            maxVal = val;
                            maxIdx = k;
                        }
                    }

                    maxValues[i][j] = maxVal;
                    maxIndices[i][j] = maxIdx;
                }
            }

            return new ValueTuplePairArray(maxIndices, maxValues);

        }

        private List<string> ReadCharacterFile()
        {
            if (!File.Exists(RecConfig.RecKeysPath))
            {
                return new List<string>();
            }
            List<string> charList = File.ReadLines(RecConfig.RecKeysPath, Encoding.UTF8).ToList();

            if (charList.Count > 0)
            {
                if (charList[charList.Count - 1].Trim() != string.Empty)
                {
                    charList.Add(" ");
                }
                else
                {
                    charList.Add(" ");
                }

            }
            charList.Insert(0, "blank");

            return charList;
        }

        public int[] getIgnoredTokens()
        {
            return [0];//for ctc blank
        }
    }
}
