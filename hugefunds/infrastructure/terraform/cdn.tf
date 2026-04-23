# HUGEFUNDS - Enterprise CDN & Edge Infrastructure
# Top 1% Grade: CloudFront with WAF, Edge Caching, DDoS Protection

# ═══════════════════════════════════════════════════════════════════════════════
# CLOUDFRONT DISTRIBUTION (Global CDN)
# ═══════════════════════════════════════════════════════════════════════════════

resource "aws_cloudfront_distribution" "main" {
  enabled             = true
  is_ipv6_enabled     = true
  comment             = "HugeFunds ${var.environment} CDN"
  default_root_object = "index.html"
  price_class         = "PriceClass_All"  # Global edge locations
  wait_for_deployment = false
  
  aliases = [var.domain_name, "*.${var.domain_name}"]

  # Origin: ALB (Dynamic Content)
  origin {
    domain_name = aws_lb.main.dns_name
    origin_id   = "ALB-${var.environment}"

    custom_origin_config {
      http_port              = 80
      https_port             = 443
      origin_protocol_policy = "https-only"
      origin_ssl_protocols   = ["TLSv1.2"]
    }

    custom_header {
      name  = "X-Origin-Verify"
      value = random_id.origin_verify.hex
    }
  }

  # Origin: S3 (Static Assets)
  origin {
    domain_name = aws_s3_bucket.static.bucket_regional_domain_name
    origin_id   = "S3-${var.environment}"
    origin_path = "/static"

    s3_origin_config {
      origin_access_identity = aws_cloudfront_origin_access_identity.main.cloudfront_access_identity_path
    }
  }

  # Default Cache Behavior (Dynamic Content)
  default_cache_behavior {
    allowed_methods  = ["DELETE", "GET", "HEAD", "OPTIONS", "PATCH", "POST", "PUT"]
    cached_methods   = ["GET", "HEAD", "OPTIONS"]
    target_origin_id = "ALB-${var.environment}"

    forwarded_values {
      query_string = true
      headers      = ["Origin", "Access-Control-Request-Headers", "Access-Control-Request-Method"]
      
      cookies {
        forward = "whitelist"
        whitelisted_names = ["session", "auth"]
      }
    }

    viewer_protocol_policy = "redirect-to-https"
    min_ttl                = 0
    default_ttl            = 0
    max_ttl                = 86400
    compress               = true

    lambda_function_association {
      event_type   = "viewer-request"
      lambda_arn   = aws_lambda_function.edge_security.qualified_arn
      include_body = false
    }
  }

  # Ordered Cache Behavior: API (No Cache)
  ordered_cache_behavior {
    path_pattern     = "/api/*"
    allowed_methods  = ["DELETE", "GET", "HEAD", "OPTIONS", "PATCH", "POST", "PUT"]
    cached_methods   = ["GET", "HEAD"]
    target_origin_id = "ALB-${var.environment}"

    forwarded_values {
      query_string = true
      headers      = ["*"]
      
      cookies {
        forward = "all"
      }
    }

    viewer_protocol_policy = "https-only"
    min_ttl                = 0
    default_ttl            = 0
    max_ttl                = 0
  }

  # Ordered Cache Behavior: Static Assets (Aggressive Caching)
  ordered_cache_behavior {
    path_pattern     = "/static/*"
    allowed_methods  = ["GET", "HEAD", "OPTIONS"]
    cached_methods   = ["GET", "HEAD"]
    target_origin_id = "S3-${var.environment}"

    forwarded_values {
      query_string = false
      cookies {
        forward = "none"
      }
    }

    viewer_protocol_policy = "redirect-to-https"
    min_ttl                = 86400        # 1 day
    default_ttl            = 604800       # 1 week
    max_ttl                = 31536000     # 1 year
    compress               = true
  }

  # Ordered Cache Behavior: WebSocket (Passthrough)
  ordered_cache_behavior {
    path_pattern     = "/ws"
    allowed_methods  = ["GET", "HEAD", "OPTIONS"]
    cached_methods   = ["GET", "HEAD"]
    target_origin_id = "ALB-${var.environment}"

    forwarded_values {
      query_string = true
      headers      = ["*"]
      cookies {
        forward = "all"
      }
    }

    viewer_protocol_policy = "https-only"
    min_ttl                = 0
    default_ttl            = 0
    max_ttl                = 0
  }

  # SSL/TLS Configuration
  viewer_certificate {
    acm_certificate_arn      = aws_acm_certificate.cloudfront.arn
    ssl_support_method       = "sni-only"
    minimum_protocol_version = "TLSv1.2_2021"
  }

  # Geo Restrictions (if needed for compliance)
  # restrictions {
  #   geo_restriction {
  #     restriction_type = "blacklist"
  #     locations        = ["IR", "KP", "SY"]
  #   }
  # }

  # Logging
  logging_config {
    include_cookies = false
    bucket          = aws_s3_bucket.logs.bucket_domain_name
    prefix          = "cdn/"
  }

  # WAF Integration
  web_acl_id = aws_wafv2_web_acl.cloudfront.arn

  # Origin Group for Failover
  origin_group {
    origin_id = "OriginGroup-${var.environment}"

    failover_criteria {
      status_codes = [500, 502, 503, 504]
    }

    member {
      origin_id = "ALB-${var.environment}"
    }

    member {
      origin_id = "S3-${var.environment}-failover"
    }
  }

  tags = local.common_tags
}

# ═══════════════════════════════════════════════════════════════════════════════
# CLOUDFRONT ORIGIN ACCESS IDENTITY
# ═══════════════════════════════════════════════════════════════════════════════

resource "aws_cloudfront_origin_access_identity" "main" {
  comment = "HugeFunds ${var.environment} OAI"
}

# ═══════════════════════════════════════════════════════════════════════════════
# ACM CERTIFICATE FOR CLOUDFRONT (us-east-1 required)
# ═══════════════════════════════════════════════════════════════════════════════

resource "aws_acm_certificate" "cloudfront" {
  provider = aws.us-east-1  # CloudFront requires cert in us-east-1

  domain_name               = var.domain_name
  subject_alternative_names = ["*.${var.domain_name}"]
  validation_method         = "DNS"

  lifecycle {
    create_before_destroy = true
  }

  tags = local.common_tags
}

# Certificate validation
resource "aws_route53_record" "cert_validation" {
  for_each = {
    for dvo in aws_acm_certificate.cloudfront.domain_validation_options : dvo.domain_name => {
      name   = dvo.resource_record_name
      record = dvo.resource_record_value
      type   = dvo.resource_record_type
    }
  }

  allow_overwrite = true
  name            = each.value.name
  records         = [each.value.record]
  ttl             = 60
  type            = each.value.type
  zone_id         = aws_route53_zone.main.zone_id
}

resource "aws_acm_certificate_validation" "cloudfront" {
  provider = aws.us-east-1

  certificate_arn         = aws_acm_certificate.cloudfront.arn
  validation_record_fqdns = [for record in aws_route53_record.cert_validation : record.fqdn]
}

# ═══════════════════════════════════════════════════════════════════════════════
# WAF WEB ACL FOR CLOUDFRONT (Edge Security)
# ═══════════════════════════════════════════════════════════════════════════════

resource "aws_wafv2_web_acl" "cloudfront" {
  provider = aws.us-east-1  # CloudFront WAF must be in us-east-1

  name        = "hugefunds-cdn-${var.environment}"
  description = "WAF rules for CloudFront"
  scope       = "CLOUDFRONT"

  default_action {
    allow {}
  }

  # AWS Managed Rules - Common Rule Set
  rule {
    name     = "AWSManagedRulesCommonRuleSet"
    priority = 1

    override_action {
      none {}
    }

    statement {
      managed_rule_group_statement {
        name        = "AWSManagedRulesCommonRuleSet"
        vendor_name = "AWS"
        
        rule_action_override {
          action_to_use {
            count {}
          }
          name = "SizeRestrictions_BODY"
        }
      }
    }

    visibility_config {
      cloudwatch_metrics_enabled = true
      metric_name                = "AWSManagedRulesCommonRuleSetMetric"
      sampled_requests_enabled   = true
    }
  }

  # AWS Managed Rules - Known Bad Inputs
  rule {
    name     = "AWSManagedRulesKnownBadInputsRuleSet"
    priority = 2

    override_action {
      none {}
    }

    statement {
      managed_rule_group_statement {
        name        = "AWSManagedRulesKnownBadInputsRuleSet"
        vendor_name = "AWS"
      }
    }

    visibility_config {
      cloudwatch_metrics_enabled = true
      metric_name                = "AWSManagedRulesKnownBadInputsRuleSetMetric"
      sampled_requests_enabled   = true
    }
  }

  # AWS Managed Rules - SQL Injection
  rule {
    name     = "AWSManagedRulesSQLiRuleSet"
    priority = 3

    override_action {
      none {}
    }

    statement {
      managed_rule_group_statement {
        name        = "AWSManagedRulesSQLiRuleSet"
        vendor_name = "AWS"
      }
    }

    visibility_config {
      cloudwatch_metrics_enabled = true
      metric_name                = "AWSManagedRulesSQLiRuleSetMetric"
      sampled_requests_enabled   = true
    }
  }

  # AWS Managed Rules - Linux OS
  rule {
    name     = "AWSManagedRulesLinuxOSRuleSet"
    priority = 4

    override_action {
      none {}
    }

    statement {
      managed_rule_group_statement {
        name        = "AWSManagedRulesLinuxOSRuleSet"
        vendor_name = "AWS"
      }
    }

    visibility_config {
      cloudwatch_metrics_enabled = true
      metric_name                = "AWSManagedRulesLinuxOSRuleSetMetric"
      sampled_requests_enabled   = true
    }
  }

  # AWS Managed Rules - POSIX
  rule {
    name     = "AWSManagedRulesPOSIXRuleSet"
    priority = 5

    override_action {
      none {}
    }

    statement {
      managed_rule_group_statement {
        name        = "AWSManagedRulesPOSIXRuleSet"
        vendor_name = "AWS"
      }
    }

    visibility_config {
      cloudwatch_metrics_enabled = true
      metric_name                = "AWSManagedRulesPOSIXRuleSetMetric"
      sampled_requests_enabled   = true
    }
  }

  # AWS Managed Rules - WordPress (if applicable)
  rule {
    name     = "AWSManagedRulesWordPressRuleSet"
    priority = 6

    override_action {
      none {}
    }

    statement {
      managed_rule_group_statement {
        name        = "AWSManagedRulesWordPressRuleSet"
        vendor_name = "AWS"
      }
    }

    visibility_config {
      cloudwatch_metrics_enabled = true
      metric_name                = "AWSManagedRulesWordPressRuleSetMetric"
      sampled_requests_enabled   = true
    }
  }

  # AWS Managed Rules - IP Reputation
  rule {
    name     = "AWSManagedRulesAmazonIpReputationList"
    priority = 7

    override_action {
      none {}
    }

    statement {
      managed_rule_group_statement {
        name        = "AWSManagedRulesAmazonIpReputationList"
        vendor_name = "AWS"
      }
    }

    visibility_config {
      cloudwatch_metrics_enabled = true
      metric_name                = "AWSManagedRulesAmazonIpReputationListMetric"
      sampled_requests_enabled   = true
    }
  }

  # AWS Managed Rules - Bot Control
  rule {
    name     = "AWSManagedRulesBotControlRuleSet"
    priority = 8

    override_action {
      none {}
    }

    statement {
      managed_rule_group_statement {
        name        = "AWSManagedRulesBotControlRuleSet"
        vendor_name = "AWS"
        
        managed_rule_group_configs {
          aws_managed_rules_bot_control_rule_set {
            inspection_level = "TARGETED"
          }
        }
      }
    }

    visibility_config {
      cloudwatch_metrics_enabled = true
      metric_name                = "AWSManagedRulesBotControlRuleSetMetric"
      sampled_requests_enabled   = true
    }
  }

  # Rate Limiting - API
  rule {
    name     = "RateLimitAPI"
    priority = 9

    action {
      block {}
    }

    statement {
      rate_based_statement {
        limit              = 1000  # requests per 5 minutes per IP
        aggregate_key_type = "IP"
        
        scope_down_statement {
          regex_pattern_set_reference_statement {
            arn = aws_wafv2_regex_pattern_set.api_paths.arn
            
            field_to_match {
              uri_path {}
            }
            
            text_transformation {
              priority = 0
              type     = "LOWERCASE"
            }
          }
        }
      }
    }

    visibility_config {
      cloudwatch_metrics_enabled = true
      metric_name                = "RateLimitAPIMetric"
      sampled_requests_enabled   = true
    }
  }

  # IP Whitelist (for admin access)
  rule {
    name     = "IPWhitelist"
    priority = 10

    action {
      allow {}
    }

    statement {
      ip_set_reference_statement {
        arn = aws_wafv2_ip_set.whitelist.arn
      }
    }

    visibility_config {
      cloudwatch_metrics_enabled = true
      metric_name                = "IPWhitelistMetric"
      sampled_requests_enabled   = true
    }
  }

  # Geographic Restriction
  rule {
    name     = "GeoBlock"
    priority = 11

    action {
      block {}
    }

    statement {
      geo_match_statement {
        country_codes = ["IR", "KP", "SY", "CU", "RU"]  # Sanctioned countries
      }
    }

    visibility_config {
      cloudwatch_metrics_enabled = true
      metric_name                = "GeoBlockMetric"
      sampled_requests_enabled   = true
    }
  }

  visibility_config {
    cloudwatch_metrics_enabled = true
    metric_name                = "hugefunds-cdn-waf"
    sampled_requests_enabled   = true
  }

  tags = local.common_tags
}

# ═══════════════════════════════════════════════════════════════════════════════
# WAF SUPPORTING RESOURCES
# ═══════════════════════════════════════════════════════════════════════════════

resource "aws_wafv2_regex_pattern_set" "api_paths" {
  provider = aws.us-east-1

  name        = "hugefunds-api-paths"
  description = "Regex patterns for API rate limiting"
  scope       = "CLOUDFRONT"

  regular_expression {
    regex_string = "^/api/.*"
  }

  tags = local.common_tags
}

resource "aws_wafv2_ip_set" "whitelist" {
  provider = aws.us-east-1

  name               = "hugefunds-admin-whitelist"
  description        = "Admin IP whitelist"
  scope              = "CLOUDFRONT"
  ip_address_version = "IPV4"
  
  addresses = var.admin_whitelist_ips

  tags = local.common_tags
}

# ═══════════════════════════════════════════════════════════════════════════════
# ROUTE 53 DNS
# ═══════════════════════════════════════════════════════════════════════════════

resource "aws_route53_zone" "main" {
  name = var.domain_name

  tags = local.common_tags
}

resource "aws_route53_record" "main" {
  zone_id = aws_route53_zone.main.zone_id
  name    = var.domain_name
  type    = "A"

  alias {
    name                   = aws_cloudfront_distribution.main.domain_name
    zone_id                = aws_cloudfront_distribution.main.hosted_zone_id
    evaluate_target_health = false
  }
}

resource "aws_route53_record" "wildcard" {
  zone_id = aws_route53_zone.main.zone_id
  name    = "*.${var.domain_name}"
  type    = "A"

  alias {
    name                   = aws_cloudfront_distribution.main.domain_name
    zone_id                = aws_cloudfront_distribution.main.hosted_zone_id
    evaluate_target_health = false
  }
}

# Health Check
resource "aws_route53_health_check" "main" {
  fqdn              = var.domain_name
  port              = 443
  type              = "HTTPS"
  resource_path     = "/api/health"
  failure_threshold = 3
  request_interval  = 30

  regions = ["us-east-1", "us-west-2", "eu-west-1"]

  tags = merge(local.common_tags, {
    Name = "hugefunds-health-check"
  })
}

# ═══════════════════════════════════════════════════════════════════════════════
# LAMBDA@EDGE (Security Headers & Request Processing)
# ═══════════════════════════════════════════════════════════════════════════════

resource "aws_iam_role" "lambda_edge" {
  name = "hugefunds-lambda-edge-${var.environment}"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = [
            "lambda.amazonaws.com",
            "edgelambda.amazonaws.com"
          ]
        }
      }
    ]
  })

  tags = local.common_tags
}

resource "aws_iam_role_policy_attachment" "lambda_edge_basic" {
  role       = aws_iam_role.lambda_edge.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}

# Lambda function for security headers
resource "aws_lambda_function" "edge_security" {
  provider = aws.us-east-1  # Lambda@Edge must be in us-east-1

  function_name = "hugefunds-edge-security-${var.environment}"
  role          = aws_iam_role.lambda_edge.arn
  handler       = "index.handler"
  runtime       = "nodejs18.x"
  publish       = true  # Required for Lambda@Edge

  filename         = data.archive_file.lambda_edge.output_path
  source_code_hash = data.archive_file.lambda_edge.output_base64sha256

  memory_size = 128
  timeout     = 3

  tags = local.common_tags
}

data "archive_file" "lambda_edge" {
  type        = "zip"
  output_path = "${path.module}/lambda_edge.zip"

  source {
    content  = <<-EOF
      exports.handler = async (event) => {
        const request = event.Records[0].cf.request;
        const response = event.Records[0].cf.response;
        
        // Add security headers
        response.headers['strict-transport-security'] = [{
          key: 'Strict-Transport-Security',
          value: 'max-age=63072000; includeSubdomains; preload'
        }];
        response.headers['x-content-type-options'] = [{
          key: 'X-Content-Type-Options',
          value: 'nosniff'
        }];
        response.headers['x-frame-options'] = [{
          key: 'X-Frame-Options',
          value: 'DENY'
        }];
        response.headers['x-xss-protection'] = [{
          key: 'X-XSS-Protection',
          value: '1; mode=block'
        }];
        response.headers['referrer-policy'] = [{
          key: 'Referrer-Policy',
          value: 'strict-origin-when-cross-origin'
        }];
        response.headers['content-security-policy'] = [{
          key: 'Content-Security-Policy',
          value: "default-src 'self'; script-src 'self' 'unsafe-inline' 'unsafe-eval'; style-src 'self' 'unsafe-inline'; img-src 'self' data: https:; font-src 'self'; connect-src 'self' https: wss:; frame-ancestors 'none'; base-uri 'self'; form-action 'self';"
        }];
        response.headers['permissions-policy'] = [{
          key: 'Permissions-Policy',
          value: 'accelerometer=(), camera=(), geolocation=(), gyroscope=(), magnetometer=(), microphone=(), payment=(), usb=()'
        }];
        
        return response;
      };
    EOF
    filename = "index.js"
  }
}

# ═══════════════════════════════════════════════════════════════════════════════
# RANDOM ID FOR ORIGIN VERIFICATION
# ═══════════════════════════════════════════════════════════════════════════════

resource "random_id" "origin_verify" {
  byte_length = 32
}

# ═══════════════════════════════════════════════════════════════════════════════
# OUTPUTS
# ═══════════════════════════════════════════════════════════════════════════════

output "cloudfront_domain_name" {
  description = "CloudFront distribution domain name"
  value       = aws_cloudfront_distribution.main.domain_name
}

output "cloudfront_distribution_id" {
  description = "CloudFront distribution ID"
  value       = aws_cloudfront_distribution.main.id
}

output "route53_zone_id" {
  description = "Route53 zone ID"
  value       = aws_route53_zone.main.zone_id
}
