var crypto = require('crypto');
var fs = require('fs');
var ALGORITHM = "sha256"; // Accepted: any result of crypto.getHashes(), check doc dor other options
var SIGNATURE_FORMAT = "base64"; // Accepted: hex, latin1, base64
const https = require('https')


function getPublicKey() {
var pubKey = fs.readFileSync('./res/public.pem', 'utf8');
 console.log("\n>>> Public key: \n\n" + pubKey);

return pubKey;
}

function getSignatureToVerify(data) {

 var signature = fs.readFileSync('./signature.sig', 'utf8');

console.log(">>> Signature:\n\n" + signature);

return signature;
}

function getData(){
    var data = fs.readFileSync('./lala.txt', 'utf8');
    return data;
}

function verifyReceivedData(recData, recSig){
  var pubkey = getPublicKey();
  var verify = crypto.createVerify(ALGORITHM);
  verify.update(recData);
  var verification = verify.verify(pubkey, recSig, SIGNATURE_FORMAT);
  return verification;
}

function isNumber(n) { return !isNaN(parseFloat(n)) && !isNaN(n - 0) }

function check_format(data){
  arr = data.split(",")
  if(arr.length != 4){
    return false;
  }
  for(var val of arr){
    if(!isNumber(val) || val > 255){
      return false;
    }
  }
  return true;
}

function updateLocalFiles(update, updateGrainy){
  var today = new Date().getDate();
  var data = fs.readFileSync('./data' + today + '.json', 'utf8');
  var grainy = fs.readFileSync('./data_grainy' + today + '.json', 'utf8');
  
  var obj = JSON.parse(data);
  var obj_g = JSON.parse(grainy);

  var u = update.split(";").slice(0,-1)
  var g = updateGrainy.split(";").slice(0,-1)

  var len_o = Object.values(obj.features).length
  var len_g = Object.values(obj_g.features).length

  var correct_lenght = (u.length == len_o) && (g.length == len_g)

  console.log(correct_lenght)

  if(!correct_lenght){
    a = 0
    // TODO
    //return -1;
  }

  // TODO Add function to check correct format of update-data
  for(let i = 0; i < len_o; i++){
    if(!check_format(u[i])){
      return -1;
    }
    obj.features[i].properties.color = "rgba(" + u[i] + ")"
  }

  for(let i = 0; i < len_g; i++){
    if(!check_format(g[i])){
      return -1;
    }
    obj_g.features[i].properties.color = "rgba(" + g[i] + ")"
  }

  var obj_string = JSON.stringify(obj);
  var obj_g_string = JSON.stringify(obj_g);

  fs.writeFileSync('./data_u.json', obj_string, 'utf-8');
  fs.writeFileSync('./data_grainy_u.json', obj_g_string, 'utf-8');
}


function handleResult(data){
  const obj = JSON.parse(data);
  var update = obj.update.data;
  var update_sig = obj.update.signature;
  var update_grainy = obj.update_grainy.data;
  var update_grainy_sig = obj.update_grainy.signature;

  var final_result = verifyReceivedData(update, update_sig) && verifyReceivedData(update_grainy, update_grainy_sig)

  console.log(final_result)
  final_result = true;
  if(final_result){
    updateLocalFiles(update, update_grainy)
  }
  else{
    console.log("SIGNATURE VERIFICATION FAILED")
  }
}

function httpGetAsync(theUrl, callback)
{
    
    const options = {
      hostname: theUrl,
      port: 443,
      path: '/data/',
      method: 'GET'
    }
    
    const req = https.request(options, res => {
      console.log(`statusCode: ${res.statusCode}`)
    
      res.on('data', d => {
        handleResult(d.toString())
      })
    })
    
    req.on('error', error => {
      console.error(error)
    })
    
    req.end()
}

function handleGetResponse(responseText){
    console.log(responseText);
}

//var publicKey = getPublicKeySomehow();
//var verify = crypto.createVerify(ALGORITHM);
//var data = getData()
//var signature = getSignatureToVerify(data);

//verify.update(data);
var data = fs.readFileSync('./update_file.json', 'utf8');

handleResult(data)
//httpGetAsync("vogel-server.de", handleGetResponse);


//var verification = verify.verify(publicKey, signature, SIGNATURE_FORMAT);

//console.log('\n>>> Verification result: ' + verification.toString().toUpperCase());