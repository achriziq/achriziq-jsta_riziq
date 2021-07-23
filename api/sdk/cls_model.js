const tf = require('@tensorflow/tfjs-node');

function normalized(data){ // i & r
    xi1 = (data[0] - -1.74075276) 
    xi2 = (data[1] - 1.35043183) 
    xi3 = (data[2] - -0.89172336) 
    xi4 = (data[3] - -1.49571784) 
    return [xi1, xi2, xi3, xi4]
}

const argFact = (compareFn) => (array)  => array.map((el,idx) => [el, idx]).reduce(compareFn)[1]
const argMax = argFact((min, el) => (el[0] > min[0] ? el:min))

function ArgMax(res){
  label = "NORMAL"
  cls_data = []
  for(i=0; i<res.length; i++){
    cls_data[i] = res[i]
  }
  console.log(cls_data,argMax(cls_data));
  
  if(argMax(cls_data) == 1){
    label  = "OVER VOLTAGE"
  }if(argMax(cls_data) == 0){
    label = "DROP VOLTAGE"
  }
  return label 
  
}


async function predict(data){
    let in_dim = 4;
    
    data = normalized(data);
    shape = [1, in_dim];

    tf_data = tf.tensor2d(data, shape);

    try{
        // path load in public access => github
        const path = 'https://raw.githubusercontent.com/achriziq/achriziq-jsta_riziq/main/public/cls_model/model.json';
        const model = await tf.loadGraphModel(path);
        
        predict = model.predict(
                tf_data
        );
        result = predict.dataSync();
        return denormalized( result );
        
    }catch(e){
      console.log(e);
    }
}

module.exports = {
    predict: predict 
}
  

