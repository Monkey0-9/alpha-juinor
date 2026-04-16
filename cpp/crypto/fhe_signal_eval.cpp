#include <iostream>
#include "seal/seal.h"

using namespace std;
using namespace seal;

/**
 * Nexus Secure Cloud Alpha Execution (Fully Homomorphic Encryption)
 * 
 * Allows the execution of proprietary quantitative alpha models on untrusted 
 * third-party cloud infrastructure. The market data is encrypted, the operations
 * (moving averages, momentum indicators) are performed on the ciphertext, 
 * and only the firm holding the private key can decrypt the trade signals.
 * 
 * Uses Microsoft SEAL implementing the CKKS scheme for approximate arithmetic.
 */

class EncryptedAlphaEngine {
public:
    EncryptedAlphaEngine() {
        EncryptionParameters parms(scheme_type::ckks);

        size_t poly_modulus_degree = 8192;
        parms.set_poly_modulus_degree(poly_modulus_degree);
        parms.set_coeff_modulus(CoeffModulus::Create(poly_modulus_degree, { 60, 40, 40, 60 }));

        context = make_shared<SEALContext>(parms);
        KeyGenerator keygen(*context);
        
        secret_key = keygen.secret_key();
        keygen.create_public_key(public_key);
        keygen.create_relin_keys(relin_keys);
        keygen.create_galois_keys(galois_keys);

        encryptor = make_unique<Encryptor>(*context, public_key);
        evaluator = make_unique<Evaluator>(*context);
        decryptor = make_unique<Decryptor>(*context, secret_key);
        encoder = make_unique<CKKSEncoder>(*context);
    }

    /**
     * @brief Evaluates an auto-regressive momentum signal completely in ciphertext.
     */
    Ciphertext compute_encrypted_momentum(const vector<Ciphertext>& encrypted_prices, double decay_factor) {
        Ciphertext result_momentum;
        Plaintext pt_decay;
        
        double scale = pow(2.0, 40);
        encoder->encode(decay_factor, scale, pt_decay);

        // Compute: M_t = M_{t-1} * decay + Price_t
        // All operations happen blindly on encrypted bytes
        evaluator->multiply_plain(encrypted_prices[0], pt_decay, result_momentum);
        evaluator->rescale_to_next_inplace(result_momentum);
        result_momentum.scale() = scale; // match scales for addition

        evaluator->add_inplace(result_momentum, encrypted_prices[1]);

        return result_momentum;
    }

private:
    shared_ptr<SEALContext> context;
    PublicKey public_key;
    SecretKey secret_key;
    RelinKeys relin_keys;
    GaloisKeys galois_keys;
    
    unique_ptr<Encryptor> encryptor;
    unique_ptr<Evaluator> evaluator;
    unique_ptr<Decryptor> decryptor;
    unique_ptr<CKKSEncoder> encoder;
};
