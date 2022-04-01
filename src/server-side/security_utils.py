from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives import serialization
import base64
from cryptography.hazmat.backends import default_backend

from Crypto.Hash import SHA256
from Crypto.PublicKey import RSA
from Crypto.Signature import pkcs1_15

from base64 import b64encode

# Code from Stackoverflow https://stackoverflow.com/questions/50608010/how-to-verify-a-signed-file-in-python
def generate_key_pair(priv_key_file, pub_key_file):
    # Generate the public/private key pair.

    private_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=4096,
        backend=default_backend(),
    )

    # Save the private key to a file.
    with open(priv_key_file, 'wb') as f:
        f.write(
            private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.TraditionalOpenSSL,
                encryption_algorithm=serialization.NoEncryption(),
            )
        )

    # Save the public key to a file.
    with open(pub_key_file, 'wb') as f:
        f.write(
            private_key.public_key().public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo,
            )
        )

def sign_file(priv_key_file, data_file, sig_file):
    # Load the private key.
    with open(priv_key_file, 'r') as f:
        private_key = RSA.import_key(f.read())
    with open(data_file, 'rb') as f:
        payload = f.read()
    # hash the message
    digest = SHA256.new(payload)

    # sign the digest
    signature = pkcs1_15.new(private_key).sign(digest)
    signature = b64encode(signature)

    with open(sig_file, 'wb') as f:
        f.write(signature)

def verify_sig(pub_key_file, data_file, signature_file):
    with open(pub_key_file, 'r') as f:
        public_key = RSA.import_key(f.read())

    with open(data_file, 'rb') as f:
        payload = f.read()

    with open(signature_file, 'rb') as f:
        signature = f.read()
    # hash the message
    digest = SHA256.new(payload)

    signature = base64.b64decode(signature)
    # verify the digest and signature
    pkcs1_15.new(public_key).verify(digest, signature)

priv_key = "./private.key"
pub_key = "./public.pem"
data = "./lala.txt"
sig_file = "./signature.sig"

#generate_key_pair(priv_key, pub_key)
sign_file(priv_key, data, sig_file)
verify_sig(pub_key, data, sig_file)