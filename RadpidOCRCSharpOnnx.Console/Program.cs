
using OpenCvSharp;
using RadpidOCRCSharpOnnx.Config;
using RadpidOCRCSharpOnnx.InferenceEngine;
using RadpidOCRCSharpOnnx.Utils;
using System.Runtime.InteropServices;
namespace RadpidOCRCSharpOnnx.ConsoleApp
{
    internal class Program
    {
        static void Main(string[] args)
        {
            var buildNumber = Environment.OSVersion.Version.Build;
            //var dd= NativeMethods.core_getVersionRevision
            Console.WriteLine("Hello, World!");

            string imgPath = @"E:\Hp\ai-image\ADFtools\latin.jpg";
            using Mat img = Cv2.ImRead(imgPath);
            (Mat ResizedImg, double RatioH, double RatioW) =UtilsHelper.ResizeImageWithinBounds(img, GlobalConfig.MinSideLen, GlobalConfig.MaxSideLen);

            (Mat ProcessedImg, int paddingTop, int paddingLeft) = UtilsHelper.ApplyVerticalPadding(ResizedImg, GlobalConfig.WidthHeightRatio, GlobalConfig.MinHeight);


            TextOCRDetector detector = new TextOCRDetector();

            var dd = detector.Run(ProcessedImg);

        }
    }
}
