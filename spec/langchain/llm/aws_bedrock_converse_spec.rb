# frozen_string_literal: true

require "aws-sdk-bedrockruntime"

RSpec.describe Langchain::LLM::AwsBedrockConverse do
  let(:subject) { described_class.new }

  before do
    stub_const("ENV", ENV.to_hash.merge("AWS_REGION" => "us-east-1"))
  end

  describe "#chat" do
    it "calls Bedrock converse and returns an AnthropicResponse" do
      # Create mocks that match the AWS SDK response structure
      text_block = double("text_block")
      allow(text_block).to receive(:text).and_return("Hello from Bedrock")
      allow(text_block).to receive(:tool_use).and_return(nil)

      output = double("output")
      allow(output).to receive(:role).and_return("assistant")
      allow(output).to receive(:content).and_return([text_block])

      usage = double("usage")
      allow(usage).to receive(:input_tokens).and_return(3)
      allow(usage).to receive(:output_tokens).and_return(5)

      response = double("response")
      allow(response).to receive(:id).and_return("msg_123")
      allow(response).to receive(:output).and_return(output)
      allow(response).to receive(:stop_reason).and_return("end_turn")
      allow(response).to receive(:usage).and_return(usage)

      expect(subject.client).to receive(:converse).with(hash_including(
        model_id: subject.defaults[:chat_model],
        messages: [{role: "user", content: [{text: "Hi"}]}]
      )).and_return(response)

      res = subject.chat(messages: [{role: "user", content: "Hi"}])

      expect(res).to be_a(Langchain::LLM::AnthropicResponse)
      expect(res.chat_completion).to eq("Hello from Bedrock")
      expect(res.prompt_tokens).to eq(3)
      expect(res.completion_tokens).to eq(5)
    end

    it "streams native events (yielded as-is) and returns an AnthropicResponse" do
      module Aws; end unless defined?(Aws)
      module Aws::BedrockRuntime; end unless defined?(Aws::BedrockRuntime)
      module Aws::BedrockRuntime::Types; end unless defined?(Aws::BedrockRuntime::Types)

      MessageStartEvent = Class.new do
        attr_reader :role
        def initialize(role:)
          @role = role
        end
        def to_h
          {role: @role}
        end
      end

      ContentBlockStartEvent = Class.new do
        attr_reader :content_block_index, :start
        def initialize(content_block_index:, start:)
          @content_block_index = content_block_index
          @start = start
        end
      end

      ContentBlockDeltaEvent = Class.new do
        attr_reader :content_block_index, :delta
        def initialize(content_block_index:, delta:)
          @content_block_index = content_block_index
          @delta = delta
        end
      end

      ConverseStreamMetadataEvent = Class.new do
        def initialize(usage: nil, stop_reason: nil)
          @usage = usage
          @stop_reason = stop_reason
        end
        def to_h
          h = {}
          h[:usage] = @usage.to_h if @usage
          h[:stop_reason] = @stop_reason if @stop_reason
          h
        end
      end

      MessageStopEvent = Class.new do
        attr_reader :stop_reason
        def initialize(stop_reason:, usage: nil)
          @stop_reason = stop_reason
          @usage = usage
        end
        def to_h
          h = {stop_reason: @stop_reason}
          h[:usage] = @usage.to_h if @usage
          h
        end
      end

      Usage = Struct.new(:input_tokens, :output_tokens) do
        def to_h
          {input_tokens: input_tokens, output_tokens: output_tokens}
        end
      end

      stub_const("Aws::BedrockRuntime::Types::MessageStartEvent", MessageStartEvent)
      stub_const("Aws::BedrockRuntime::Types::ContentBlockStartEvent", ContentBlockStartEvent)
      stub_const("Aws::BedrockRuntime::Types::ContentBlockDeltaEvent", ContentBlockDeltaEvent)
      stub_const("Aws::BedrockRuntime::Types::ConverseStreamMetadataEvent", ConverseStreamMetadataEvent)
      stub_const("Aws::BedrockRuntime::Types::MessageStopEvent", MessageStopEvent)

      start = Struct.new(:h) do
        def to_h
          h
        end
      end

      delta = Struct.new(:h) do
        def to_h
          h
        end
      end

      usage = Usage.new(3, 5)
      events = [
        Aws::BedrockRuntime::Types::MessageStartEvent.new(role: "assistant"),
        Aws::BedrockRuntime::Types::ContentBlockStartEvent.new(
          content_block_index: 0,
          start: start.new({})
        ),
        Aws::BedrockRuntime::Types::ContentBlockDeltaEvent.new(
          content_block_index: 0,
          delta: delta.new({text: "Hello"})
        ),
        Aws::BedrockRuntime::Types::MessageStopEvent.new(stop_reason: "end_turn", usage: usage)
      ]

      stream = double("stream")
      allow(stream).to receive(:on_event) do |&blk|
        events.each { |e| blk.call(e) }
      end

      expect(subject.client).to receive(:converse_stream).and_yield(stream)

      yielded = []
      res = subject.chat(messages: [{role: "user", content: "Hi"}]) do |event|
        yielded << event
      end

      expect(yielded).to eq(events)
      expect(res).to be_a(Langchain::LLM::AnthropicResponse)
      expect(res.chat_completion).to eq("Hello")
      expect(res.stop_reason).to eq("end_turn")
      expect(res.prompt_tokens).to eq(3)
      expect(res.completion_tokens).to eq(5)
    end
  end
end

