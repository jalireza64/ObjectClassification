using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Mvc;
using Microsoft.ML;
using Microsoft.ML.Data;
using System;
using System.Drawing;
using System.IO;
using Tensorflow.Keras;

namespace ObjectClassification.Controllers
{
    [ApiController]
    [Route("[controller]")]
    public class SentenceClassificationController : ControllerBase
    {
        private readonly ILogger<SentenceClassificationController> _logger;
        private IWebHostEnvironment _hostingEnvironment;

        public SentenceClassificationController(ILogger<SentenceClassificationController> logger, IWebHostEnvironment environment)
        {
            _logger = logger;
            _hostingEnvironment = environment;
        }

        [HttpPost(nameof(GetSentenceSentiment))]
        public IActionResult GetSentenceSentiment(string sentence)
        {
            //Load sample data
            var sampleData = new SentenceClassificationModel.ModelInput()
            {
                Col0 = sentence,
            };

            //Load model and predict output
            var result = SentenceClassificationModel.Predict(sampleData);
            var finalResult = new
            {
                Type = result.PredictedLabel,
                result.Score
            };

            return Ok(finalResult);
        }

        [HttpPost(nameof(TrainSentenceClassification))]
        public IActionResult TrainSentenceClassification(string sentence, bool sentenceType)
        {
            string dataModelRoot = Path.Combine(_hostingEnvironment.ContentRootPath, "DataModel\\Sentence\\SentenceData.txt");

            var insertedSample = sentenceType ? sentence + ",Positive" : sentence + ",Negative";
            System.IO.File.AppendAllText(dataModelRoot, insertedSample + "\r\n");

            MLContext mlContext = new MLContext();
            var newData = SentenceClassificationModel.LoadIDataViewFromFile(mlContext, dataModelRoot, ',', false);
            try
            {
                SentenceClassificationModel.RetrainModel(mlContext, newData);
            }
            catch (Exception ex)
            {

            }

            return Ok();
        }
    }
}
