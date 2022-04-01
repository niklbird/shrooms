var crypto = require('crypto');
var fs = require('fs');
var ALGORITHM = "sha256"; // Accepted: any result of crypto.getHashes(), check doc dor other options
var SIGNATURE_FORMAT = "base64"; // Accepted: hex, latin1, base64

function getPublicKeySomehow() {

var pubKey = fs.readFileSync('./public.pem', 'utf8');
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

var publicKey = getPublicKeySomehow();
var verify = crypto.createVerify(ALGORITHM);
var data = getData()
var signature = getSignatureToVerify(data);

verify.update(data);

var verification = verify.verify(publicKey, signature, SIGNATURE_FORMAT);

console.log('\n>>> Verification result: ' + verification.toString().toUpperCase());