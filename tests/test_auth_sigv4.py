import unittest


class TestSigV4Headers(unittest.TestCase):
    def test_aws_docs_known_answer_vector(self):
        """Invariant: the official get-vanilla case yields its exact Authorization value."""
        from core.auth_sigv4 import sigv4_headers

        out = sigv4_headers(
            method="GET",
            url="https://example.amazonaws.com/",
            headers={},
            body=b"",
            access_key="AKIDEXAMPLE",
            secret_key="wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY",
            session_token=None,
            region="us-east-1",
            service="service",
            amz_date="20150830T123600Z",
        )
        self.assertEqual(
            out["Authorization"],
            "AWS4-HMAC-SHA256 "
            "Credential=AKIDEXAMPLE/20150830/us-east-1/service/aws4_request, "
            "SignedHeaders=host;x-amz-date, "
            "Signature=5fa00fa31553b73ebf1942676e86291e8372ff2a2260956d9b8aae1d763fbf31",
        )

    def test_vector_carries_the_amz_date_header(self):
        """Invariant: signing stamps the x-amz-date header it signed."""
        from core.auth_sigv4 import sigv4_headers

        out = sigv4_headers(
            method="GET",
            url="https://iam.amazonaws.com/",
            headers={},
            body=b"",
            access_key="AKIDEXAMPLE",
            secret_key="wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY",
            session_token=None,
            region="us-east-1",
            service="iam",
            amz_date="20150830T123600Z",
        )
        self.assertEqual(out["x-amz-date"], "20150830T123600Z")

    def test_inputs_are_never_mutated(self):
        """Invariant: sigv4_headers returns a new dict and leaves inputs alone."""
        from core.auth_sigv4 import sigv4_headers

        headers = {"Content-Type": "application/json"}
        out = sigv4_headers(
            method="POST",
            url="https://bedrock-runtime.us-east-1.amazonaws.com/openai/v1/chat/completions",
            headers=headers,
            body=b"{}",
            access_key="AKID",
            secret_key="SECRET",
            session_token=None,
            region="us-east-1",
            service="bedrock",
            amz_date="20260101T000000Z",
        )
        self.assertIsNot(out, headers)
        self.assertEqual(headers, {"Content-Type": "application/json"})
        self.assertIn("Authorization", out)
        self.assertIn("Content-Type", out)

    def test_session_token_is_signed_and_sent(self):
        """Invariant: a session token adds the security-token header to the signature."""
        from core.auth_sigv4 import sigv4_headers

        out = sigv4_headers(
            method="POST",
            url="https://bedrock-runtime.us-east-1.amazonaws.com/openai/v1",
            headers={},
            body=b"{}",
            access_key="AKID",
            secret_key="SECRET",
            session_token="TOKEN",
            region="us-east-1",
            service="bedrock",
            amz_date="20260101T000000Z",
        )
        self.assertEqual(out["x-amz-security-token"], "TOKEN")
        credential_scope = out["Authorization"].split("Credential=")[1]
        self.assertTrue(credential_scope.startswith("AKID/20260101/us-east-1/bedrock/aws4_request"))

    def test_wrong_secret_changes_the_signature(self):
        """Invariant: the signature actually depends on the secret key."""
        from core.auth_sigv4 import sigv4_headers

        kwargs = dict(
            method="GET",
            url="https://example.amazonaws.com/",
            headers={},
            body=b"",
            access_key="AKIDEXAMPLE",
            session_token=None,
            region="us-east-1",
            service="service",
            amz_date="20150830T123600Z",
        )
        good = sigv4_headers(secret_key="wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY", **kwargs)
        bad = sigv4_headers(secret_key="wrong", **kwargs)
        self.assertNotEqual(good["Authorization"], bad["Authorization"])

    def test_missing_credential_raises_valueerror(self):
        """Invariant: signing without a key pair fails instead of sending garbage."""
        from core.auth_sigv4 import sigv4_headers

        with self.assertRaises(ValueError):
            sigv4_headers(
                method="GET",
                url="https://example.com/",
                headers={},
                body=b"",
                access_key="",
                secret_key="",
                session_token=None,
                region="us-east-1",
                service="iam",
                amz_date="20150830T123600Z",
            )

    def test_mixed_case_reserved_headers_are_dropped_before_stamping(self):
        """Invariant: stale Host/X-Amz-Date/X-Amz-Security-Token (any case) neither duplicate nor sign."""
        from core.auth_sigv4 import sigv4_headers

        out = sigv4_headers(
            method="GET",
            url="https://example.amazonaws.com/",
            headers={
                "Host": "evil.example.com",
                "X-AMZ-DATE": "19990101T000000Z",
                "X-Amz-Security-Token": "STALE",
            },
            body=b"",
            access_key="AKIDEXAMPLE",
            secret_key="wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY",
            session_token=None,
            region="us-east-1",
            service="service",
            amz_date="20150830T123600Z",
        )
        self.assertEqual(
            out["Authorization"],
            "AWS4-HMAC-SHA256 "
            "Credential=AKIDEXAMPLE/20150830/us-east-1/service/aws4_request, "
            "SignedHeaders=host;x-amz-date, "
            "Signature=5fa00fa31553b73ebf1942676e86291e8372ff2a2260956d9b8aae1d763fbf31",
        )
        self.assertEqual(out["host"], "example.amazonaws.com")
        self.assertEqual(out["x-amz-date"], "20150830T123600Z")
        self.assertEqual(
            set(out), {"host", "x-amz-date", "Authorization"}
        )
        self.assertNotIn("Host", out)
        self.assertNotIn("X-AMZ-DATE", out)
        self.assertNotIn("X-Amz-Security-Token", out)
        self.assertNotIn("x-amz-security-token", out)


class TestSigV4Signer(unittest.TestCase):
    def test_sign_uses_service_and_region_from_options_and_secrets(self):
        """Invariant: SigV4Signer defaults to bedrock, region from secrets."""
        from core.auth_sigv4 import SigV4Signer

        signer = SigV4Signer()
        out = signer.sign(
            method="POST",
            url="https://bedrock-runtime.us-east-1.amazonaws.com/openai/v1",
            headers={},
            body=b"{}",
            options={},
            secrets={"access_key": "AKID", "secret_key": "SECRET", "region": "us-east-1"},
        )
        self.assertTrue(out["Authorization"].startswith("AWS4-HMAC-SHA256 "))
        self.assertIn("/us-east-1/bedrock/aws4_request", out["Authorization"])

    def test_region_may_come_from_options(self):
        """Invariant: options region fills in when secrets carry no region."""
        from core.auth_sigv4 import SigV4Signer

        signer = SigV4Signer()
        out = signer.sign(
            method="POST",
            url="https://example.com/",
            headers={},
            body=b"{}",
            options={"region": "eu-west-1", "service": "execute-api"},
            secrets={"access_key": "AKID", "secret_key": "SECRET"},
        )
        self.assertIn("/eu-west-1/execute-api/aws4_request", out["Authorization"])

    def test_missing_secrets_raise_valueerror(self):
        """Invariant: SigV4Signer without credentials fails closed."""
        from core.auth_sigv4 import SigV4Signer

        signer = SigV4Signer()
        with self.assertRaises(ValueError):
            signer.sign(
                method="GET",
                url="https://example.com/",
                headers={},
                body=b"",
                options={},
                secrets={},
            )

    def test_missing_region_raises_valueerror(self):
        """Invariant: SigV4Signer without a region fails closed."""
        from core.auth_sigv4 import SigV4Signer

        signer = SigV4Signer()
        with self.assertRaises(ValueError):
            signer.sign(
                method="GET",
                url="https://example.com/",
                headers={},
                body=b"",
                options={},
                secrets={"access_key": "AKID", "secret_key": "SECRET"},
            )


if __name__ == "__main__":
    unittest.main()
