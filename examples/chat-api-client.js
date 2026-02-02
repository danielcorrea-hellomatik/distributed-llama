import util from 'node:util';

// This is a simple client for dllama-api.
//
// Usage:
//
// 1. Start the server, how to do it is described in the `src/apps/dllama-api/README.md` file.
// 2. Run this script: `node examples/chat-api-client.js`

const HOST = process.env.HOST ? process.env.HOST : '127.0.0.1';
const PORT = process.env.PORT ? Number(process.env.PORT) : 9990;
const DEBUG = ['1', 'true'].includes(process.env.DEBUG);

function debug(name, content) {
    if (DEBUG) {
        console.log(name, util.inspect(content, {
            colors: true,
            depth: null,
            compact: false,
        }));
    }
}

async function complete(messages, maxTokens, extra = {}) {
    const body = {
        messages,
        temperature: 0.7,
        stop: ['<|eot_id|>'],
        max_tokens: maxTokens,
        ...extra
    };
    debug('🔵 Request', body);
    const response = await fetch(`http://${HOST}:${PORT}/v1/chat/completions`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(body),
    });
    const json = await response.json();
    debug('🔴 Response', json);
    return json;
}

async function ask(system, user, maxTokens) {
    console.log(`> system: ${system}`);
    console.log(`> user: ${user}`);
    const response = await complete([
        {
            role: 'system',
            content: system
        },
        {
            role: 'user',
            content: user
        }
    ], maxTokens);
    console.log(response.usage);
    console.log(response.choices[0].message.content);
}

async function askWithTools(companyName, maxTokens) {
    const tools = [
        {
            type: 'function',
            function: {
                name: 'get_most_popular_car_by_company',
                description: 'Return the most popular car model for a given company name.',
                parameters: {
                    type: 'object',
                    properties: {
                        companyName: {
                            type: 'string',
                            description: 'Car company name, e.g., `Toyota`, `Ford`'
                        }
                    },
                    required: ['companyName']
                }
            }
        },
        {
            type: 'function',
            function: {
                name: 'get_car_sales_this_year',
                description: 'Return total sales for the given car company for the current calendar year.',
                parameters: {
                    type: 'object',
                    properties: {
                        companyNameAndCarName: {
                            type: 'string',
                            description: 'Car company name and car name concatenated, e.g., `Toyota - Corolla`'
                        }
                    },
                    required: ['companyNameAndCarName']
                }
            }
    }];

    const messages = [
        { role: 'system', content: 'You can use only 1 tool at the time.' },
        { role: 'user', content: `Tell me about the most popular car from ${companyName} and its sales this year.` }
    ];

    console.log(`> user: ${messages[1].content}`);

    let response;
    for (let i = 0; ; i++) {
        response = await complete(messages, maxTokens, { tools, tool_choice: 'auto' });
        const choice = response.choices[0];

        if (choice.finish_reason !== 'tool_calls' || !choice.message.tool_calls?.length) {
            break;
        }
        messages.push(choice.message);
        for (const call of choice.message.tool_calls) {
            switch (call.function.name) {
                case 'get_most_popular_car_by_company':
                    messages.push({
                        role: 'tool',
                        tool_call_id: call.id,
                        content: JSON.stringify({
                            carName: 'Corolla'
                        })
                    });
                    break;
                case 'get_car_sales_this_year':
                    messages.push({
                        role: 'tool',
                        tool_call_id: call.id,
                        content: JSON.stringify({
                            salesThisYear: 250000
                        })
                    });
                    break;
                default:
                    throw new Error(`Unsupported tool: ${call.function.name}`);
            }
        }
    }

    console.log(response.choices[0].message.content);
}

async function main() {
    await ask('You are an excellent math teacher.', 'What is 1 + 2?', 256);
    await ask('You are a romantic.', 'Where is Europe?', 256);
    await askWithTools('Toyota', 5000);
}

main();
