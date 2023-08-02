using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Mvc;
using Microsoft.ML.Data;
using System.Drawing;
using System.IO;

namespace ObjectClassification.Controllers
{
    [ApiController]
    [Route("[controller]")]
    public class ImageClassificationController : ControllerBase
    {
        private readonly ILogger<ImageClassificationController> _logger;

        public ImageClassificationController(ILogger<ImageClassificationController> logger)
        {
            _logger = logger;
        }

        [HttpPost(nameof(GetAnimalDetection))]
        public IActionResult GetAnimalDetection(IFormFile file)
        {

            
            MLImage image;
            if (file == null || file.Length == 0)
            {
                return BadRequest();
            }

            using (var memoryStream = new MemoryStream())
            {
                file.CopyTo(memoryStream);
                var base64 = Convert.ToBase64String(memoryStream.ToArray());

                var outputStream = new MemoryStream(Convert.FromBase64String(base64));


                image = MLImage.CreateFromStream(outputStream);
            }
            
            ObjectClassificationModel.ModelInput sampleData = new ObjectClassificationModel.ModelInput()
            {
                Image = image,
            };

            var result = ObjectClassificationModel.Predict(sampleData);

            var finalResult = new
            {
                Animal = result.PredictedLabel,
                Score = result.Score
            };

            return Ok(finalResult);
        }

        [HttpPost(nameof(GetMaskDetection))]
        public IActionResult GetMaskDetection(IFormFile file)
        {
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

            return Ok(finalResult);
        }
    }
}
