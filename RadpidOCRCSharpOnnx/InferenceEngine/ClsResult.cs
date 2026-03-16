using System;
using System.Collections.Generic;
using System.Text;

namespace RadpidOCRCSharpOnnx.InferenceEngine
{
    public struct ClsResult
    {
        public string Label { get; private set; }
        public float Score { get; private set; }

        public ClsResult(string label, float score)
        {
            Label = label;
            Score = score;
        }
    }
}
