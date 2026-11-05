package main

import (
	"fmt"
	"log"
	"net/http"

	"github.com/gin-contrib/cors"
	"github.com/gin-gonic/gin"
)

func main() {
	startRouter()
}

func startRouter() {
	r := gin.Default()
	r.Use(cors.Default())
	r.StaticFile("/", "./public/index.html")
	r.Static("/assets", "./public")

	r.GET("/ping", func(c *gin.Context) {
		// Return JSON response
		c.JSON(http.StatusOK, gin.H{
			"message": "pong",
		})
	})

	r.POST("/guess", guess)

	if err := r.Run(); err != nil {
		log.Fatalf("failed to run server: %v", err)
	}
}

func guess(c *gin.Context) {
	var clientDataStruct struct {
		Bytes []float64 `json:"bytes"`
	}

	if err := c.ShouldBindBodyWithJSON(&clientDataStruct); err != nil {
		fmt.Printf("Failed to parse data: %e", err)
		c.JSON(400, "ERROR")
		return
	}

	var inputs = NewMatrix(784, 1)
	for i := 0; i < 784; i++ {
		inputs[i][0] = clientDataStruct.Bytes[i]
	}

	output := calculate(inputs)
	c.JSON(200, gin.H{"results": output})
}
