using System;
using System.Collections.Generic;
using System.Text;

namespace RadpidOCRCSharpOnnx.InferenceEngine
{
    public struct ValueTuplePairArray
    {
        public int[][] Indices { get; set; }
        public float[][] Values { get; set; }

        public ValueTuplePairArray(int[][] indices, float[][] values)
        {
            Indices = indices;
            Values = values;
        }
    }
}
