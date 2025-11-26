import { describe, it, expect, beforeEach, vi, afterEach } from "vitest"
import { VertexHandler } from "../vertex"
import { ApiHandlerOptions } from "../../../shared/api"

// Mock i18next
vi.mock("i18next", () => ({
	t: (key: string) => key,
}))

describe("VertexHandler Express Mode", () => {
	let handler: VertexHandler
	let fetchMock: any

	const options: ApiHandlerOptions = {
		apiModelId: "gemini-2.0-flash-001",
		vertexApiKey: "test-api-key",
		vertexProjectId: undefined,
		vertexRegion: undefined,
	}

	beforeEach(() => {
		fetchMock = vi.fn()
		global.fetch = fetchMock
		handler = new VertexHandler(options)
	})

	afterEach(() => {
		vi.restoreAllMocks()
	})

	it("should use correct generation config with defaults", async () => {
		const stream = new ReadableStream({
			start(controller) {
				controller.close()
			},
		})
		const response = {
			ok: true,
			body: stream,
			text: () => Promise.resolve(""),
		}
		fetchMock.mockResolvedValue(response)

		const generator = handler.createMessage("system prompt", [{ role: "user", content: "hello" }])
		await generator.next()

		expect(fetchMock).toHaveBeenCalledWith(
			expect.stringContaining("streamGenerateContent?key=test-api-key"),
			expect.objectContaining({
				method: "POST",
			}),
		)

		const callArgs = JSON.parse(fetchMock.mock.calls[0][1].body)
		// Default temperature for Gemini is 1
		expect(callArgs.generationConfig.temperature).toBe(1)
		// Default maxOutputTokens is 8192
		expect(callArgs.generationConfig.maxOutputTokens).toBe(8192)
	})

	it("should use user provided configuration", async () => {
		const customOptions = { ...options, modelTemperature: 0.5, modelMaxTokens: 1000 }
		handler = new VertexHandler(customOptions)

		const stream = new ReadableStream({
			start(controller) {
				controller.close()
			},
		})
		const response = {
			ok: true,
			body: stream,
			text: () => Promise.resolve(""),
		}
		fetchMock.mockResolvedValue(response)

		const generator = handler.createMessage("system prompt", [{ role: "user", content: "hello" }])
		await generator.next()

		const callArgs = JSON.parse(fetchMock.mock.calls[0][1].body)
		expect(callArgs.generationConfig.temperature).toBe(0.5)
		expect(callArgs.generationConfig.maxOutputTokens).toBe(1000)
	})

	it("should parse streaming JSON correctly", async () => {
		const chunks = [
			'{"candidates": [{"content": {"parts": [{"text": "Hello"}]}}]}',
			'{"candidates": [{"content": {"parts": [{"text": " world"}]}}]}',
			'{"usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 5}}',
		]

		const encoder = new TextEncoder()
		const stream = new ReadableStream({
			start(controller) {
				chunks.forEach((chunk) => controller.enqueue(encoder.encode(chunk)))
				controller.close()
			},
		})

		fetchMock.mockResolvedValue({
			ok: true,
			body: stream,
			text: () => Promise.resolve(""),
		})

		const messages = []
		for await (const msg of handler.createMessage("sys", [])) {
			messages.push(msg)
		}

		expect(messages).toEqual([
			{ type: "text", text: "Hello" },
			{ type: "text", text: " world" },
			{ type: "usage", inputTokens: 10, outputTokens: 5, totalCost: expect.any(Number) },
		])
	})

	it("should handle split JSON chunks", async () => {
		const chunks = [
			'{"candidates": [{"content": {"parts": [{"text": "Split"}]}}',
			"]}",
			'{"candidates": [{"content": {"parts": [{"text": " "}]}}]}',
			'{"candidates": [{"content": {"parts": [{"text": "Chunk"}]}}]}',
		]

		const encoder = new TextEncoder()
		const stream = new ReadableStream({
			start(controller) {
				chunks.forEach((chunk) => controller.enqueue(encoder.encode(chunk)))
				controller.close()
			},
		})

		fetchMock.mockResolvedValue({
			ok: true,
			body: stream,
			text: () => Promise.resolve(""),
		})

		const messages = []
		for await (const msg of handler.createMessage("sys", [])) {
			messages.push(msg)
		}

		expect(messages).toEqual([
			{ type: "text", text: "Split" },
			{ type: "text", text: " " },
			{ type: "text", text: "Chunk" },
		])
	})
})
