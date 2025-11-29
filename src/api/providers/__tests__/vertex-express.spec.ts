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
		expect(callArgs.generationConfig.temperature).toBe(1)
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

	it("should correctly format complex messages using convertAnthropicMessageToGemini", async () => {
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

		// Create a message with tool use structure that would have been JSON-stringified naively
		const messages = [
			{
				role: "assistant",
				content: [
					{ type: "text", text: "I will run this." },
					{ type: "tool_use", id: "call_1", name: "execute_command", input: { command: "ls" } },
				],
			},
		]

		// Mock toolIdToName logic if needed, but here we just want to see the request body structure
		const generator = handler.createMessage("system prompt", messages as any)
		await generator.next()

		const callArgs = JSON.parse(fetchMock.mock.calls[0][1].body)
		const contents = callArgs.contents

		// Should be a proper structure, NOT a JSON string in 'text'
		expect(contents[0].parts[0]).toHaveProperty("text", "I will run this.")
		expect(contents[0].parts[1]).toHaveProperty("functionCall")
		expect(contents[0].parts[1].functionCall).toEqual({
			name: "execute_command",
			args: { command: "ls" },
		})
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

	it("should parse <think> tags and emit reasoning events", async () => {
		const chunks = [
			'{"candidates": [{"content": {"parts": [{"text": "<think>This is a thought"}]}}]}',
			'{"candidates": [{"content": {"parts": [{"text": " still thinking</think>"}]}}]}',
			'{"candidates": [{"content": {"parts": [{"text": "This is the response"}]}}]}',
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
			{ type: "reasoning", text: "This is a thought" },
			{ type: "reasoning", text: " still thinking" },
			{ type: "text", text: "This is the response" },
		])
	})

	it("should include tools in request and clean schema parameters", async () => {
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

		const tools = [
			{
				name: "test_tool",
				description: "A test tool",
				parameters: {
					type: "object",
					properties: {
						param1: {
							type: "string",
							description: "Parameter 1",
							default: "default_value",
						},
						param2: {
							type: "integer",
							minimum: 0,
							maximum: 10,
							exclusiveMaximum: true,
						},
						param3: {
							type: ["string", "null"],
							description: "A nullable string",
						},
					},
					required: ["param1"],
					additionalProperties: false,
				},
			},
		]

		// Pass tools in metadata
		const generator = handler.createMessage("system prompt", [], {
			tools: tools.map((t) => ({
				type: "function",
				function: {
					name: t.name,
					description: t.description,
					parameters: t.parameters,
				},
			})),
		})
		await generator.next()

		const callArgs = JSON.parse(fetchMock.mock.calls[0][1].body)
		const sentTools = callArgs.tools

		expect(sentTools).toHaveLength(1)
		expect(sentTools[0].functionDeclarations).toHaveLength(1)

		const decl = sentTools[0].functionDeclarations[0]
		expect(decl.name).toBe("test_tool")
		expect(decl.description).toBe("A test tool")

		// Verify schema cleaning
		const params = decl.parameters
		expect(params).not.toHaveProperty("additionalProperties")
		expect(params.properties.param1).not.toHaveProperty("default")
		expect(params.properties.param2).not.toHaveProperty("exclusiveMaximum")
		expect(params.properties.param2).not.toHaveProperty("minimum")
		expect(params.properties.param2).not.toHaveProperty("maximum")

		// Verify preserved fields
		expect(params.properties.param1.type).toBe("string")
		expect(params.properties.param1.description).toBe("Parameter 1")
		expect(params.properties.param2.type).toBe("integer")
		expect(params.required).toEqual(["param1"])

		// Verify array type flattening (nullable handling)
		expect(params.properties.param3.type).toBe("string")
		expect(params.properties.param3.nullable).toBe(true)
	})
})
