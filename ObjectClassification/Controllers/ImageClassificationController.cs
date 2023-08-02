using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Mvc;
using Microsoft.ML;
using Microsoft.ML.Data;
using System;
using System.Drawing;
using System.IO;

namespace ObjectClassification.Controllers
{
    [ApiController]
    [Route("[controller]")]
    public class ImageClassificationController : ControllerBase
    {
        private readonly ILogger<ImageClassificationController> _logger;
        private IWebHostEnvironment _hostingEnvironment;

        public ImageClassificationController(ILogger<ImageClassificationController> logger, IWebHostEnvironment environment)
        {
            _logger = logger;
            _hostingEnvironment = environment;
        }

        [HttpPost(nameof(GetMaskDetection))]
        public IActionResult GetMaskDetection(IFormFile file)
        {
            string ImagesRoot = Path.Combine(_hostingEnvironment.ContentRootPath, "Images\\MaskDetectionSampleImages");

            byte[] outputImageByte;
            if (file == null || file.Length == 0)
            {
                return BadRequest();
            }

            using (var memoryStream = new MemoryStream())
            {
                
                file.CopyTo(memoryStream);
                var base64 = Convert.ToBase64String(memoryStream.ToArray());

                outputImageByte = Convert.FromBase64String(base64);
            }

            MaskClassificationImageModel.ModelInput sampleData = new MaskClassificationImageModel.ModelInput()
            {
                ImageSource = outputImageByte,
            };

            var result = MaskClassificationImageModel.Predict(sampleData);



            var finalResult = new
            {
                Type = result.PredictedLabel,
                Score = result.Score
            };

            if (finalResult.Type.Equals("WithMask"))
            {
                if (file.Length > 0)
                {
                    string filePath = Path.Combine(ImagesRoot, "WithMask/" + file.FileName);
                    using (Stream fileStream = new FileStream(filePath, FileMode.Create))
                    {
                        file.CopyTo(fileStream);
                    }
                }
            }
            else if(finalResult.Type.Equals("WithoutMask"))
            {
                if (file.Length > 0)
                {
                    string filePath = Path.Combine(ImagesRoot, "WithoutMask\\" + file.FileName);
                    using (Stream fileStream = new FileStream(filePath, FileMode.Create))
                    {
                        file.CopyTo(fileStream);
                    }
                }
            }

            //MLContext mlContext = new MLContext();
            //var newData = MaskClassificationImageModel.LoadImageFromFolder(mlContext, ImagesRoot);
            //try
            //{
            //    MaskClassificationImageModel.RetrainModel(mlContext, newData);
            //}
            //catch (Exception ex)
            //{

            //}


            return Ok(finalResult);
        }

        [HttpPost(nameof(TrainMaskClassification))]
        public IActionResult TrainMaskClassification()
        {
            string ImagesRoot = Path.Combine(_hostingEnvironment.ContentRootPath, "Images\\MaskDetectionSampleImages");

            MLContext mlContext = new MLContext();
            var newData = MaskClassificationImageModel.LoadImageFromFolder(mlContext, ImagesRoot);
            try
            {
                MaskClassificationImageModel.RetrainModel(mlContext, newData);
            }
            catch (Exception ex)
            {

            }

            return Ok();
        }
    }
}
