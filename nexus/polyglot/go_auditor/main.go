package main

import (
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"sync"
	"time"
)

type AuditResult struct {
	Service string  `json:"service"`
	Status  string  `json:"status"`
	Latency float64 `json:"latency_ms"`
	Error   string  `json:"error,omitempty"`
}

type PlatformReport struct {
	Timestamp string        `json:"timestamp"`
	Results   []AuditResult `json:"results"`
	Overall   string        `json:"overall_health"`
}

func checkService(service string, url string, wg *sync.WaitGroup, results chan<- AuditResult) {
	defer wg.Done()

	start := time.Now()
	client := http.Client{Timeout: 2 * time.Second}
	resp, err := client.Get(url)
	latency := float64(time.Since(start).Microseconds()) / 1000.0

	result := AuditResult{
		Service: service,
		Latency: latency,
	}

	if err != nil {
		result.Status = "DOWN"
		result.Error = err.Error()
	} else {
		defer resp.Body.Close()
		if resp.StatusCode == 200 {
			result.Status = "UP"
		} else {
			result.Status = "UNHEALTHY"
			result.Error = fmt.Sprintf("HTTP %d", resp.StatusCode)
		}
	}

	results <- result
}

func main() {
	baseURL := "http://127.0.0.1:8000"
	if len(os.Args) > 1 {
		baseURL = os.Args[1]
	}

	services := map[string]string{
		"Core API": baseURL + "/api/alpaca/health",
		"Engine":   baseURL + "/api/health",
	}

	var wg sync.WaitGroup
	results := make(chan AuditResult, len(services))

	for name, url := range services {
		wg.Add(1)
		go checkService(name, url, &wg, results)
	}

	wg.Wait()
	close(results)

	var report PlatformReport
	report.Timestamp = time.Now().Format(time.RFC3339)
	report.Overall = "STABLE"

	for res := range results {
		report.Results = append(report.Results, res)
		if res.Status != "UP" {
			report.Overall = "DEGRADED"
		}
	}

	output, _ := json.MarshalIndent(report, "", "  ")
	fmt.Println(string(output))
}
