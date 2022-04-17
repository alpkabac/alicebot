import {config} from "dotenv";

config()
import channelBotTranslationService from "./personalityService.js";
import historyService from "./historyService.js";
import memoryService from "./memoryService.js";
import encoder from "gpt-3-encoder";


class PromptService {
    static getIntroduction(botTranslations, usesIntroduction = true, privateMessage = false) {
        if (!usesIntroduction) return []
        return (
            privateMessage ?
                botTranslations.introductionDm
                : botTranslations.introduction
        )
            .map((e) => {
                return {
                    from: e.from.replace("${botName}", process.env.BOTNAME),
                    msg: e.msg.replace("${botName}", process.env.BOTNAME)
                }
            })
    }

    static getChannelMemory(channel, usesMemory = true) {
        if (!usesMemory) return []
        return Object.keys(memoryService.getChannelMemory(channel))
            .map((key) => {
                return {from: key, msg: memoryService.getChannelMemory(channel)[key]}
            })
    }

    static mapJoinMessages(messages) {
        return messages
            .map((msg) => msg.from ? `${msg.from}: ${msg.msg}` : msg.msg)
            .join("\n")
    }

    static getNoContextPrompt(msg, from, channel) {
        const botTranslations = channelBotTranslationService.getChannelPersonality(channel)

        return PromptService.mapJoinMessages([
            {from: process.env.BOTNAME, msg: botTranslations.noContextSentence},
            {from, msg}
        ]) + "\n" + process.env.BOTNAME + ":"
    }

    static getPrompt(channel, isContinuation = false, isRetry = false, historyEnabled = true) {
        const privateConversation = channel.startsWith("##")
        const botTranslations = channelBotTranslationService.getChannelPersonality(channel)
        const channelContext = privateConversation ?
            botTranslations.contextDm
            : botTranslations.context
        const botDescription = botTranslations.description

        const channelMemory = this.getChannelMemory(channel)
            .map(m => m.msg).join("\n")         // Insert channel `!remember`s

        const introduction = this.getIntroduction(botTranslations, true, privateConversation)

        const history = JSON.parse(JSON.stringify(
            historyService
                .getChannelHistory(channel)
        ))

        let promptContext = ""
        if (channelContext) {
            promptContext += channelContext + '\n'
        }
        if (botDescription) {
            promptContext += botDescription + '\n'
        }
        if (channelMemory) {
            promptContext += channelMemory + '\n'
        }

        const minimalContextLength = encoder.encode(promptContext).length
        if (introduction) {
            promptContext += PromptService.mapJoinMessages(introduction) + '\n'
            console.log("promptContext:" + promptContext) 
        }

        const contextLength = encoder.encode(promptContext).length
        const lastLine = process.env.BOTNAME + ":"
        const lastLineLength = encoder.encode(lastLine).length

        // Inserts as much history as possible in the 2048 token limits (including context and last line)
        let promptHistory = ""
        let couldInsertAllHistory = true
        if (historyEnabled) {
            let lastBotMessageFound = false
            for (let i = history.length - 1; i >= 0; i--) {

                if ((isRetry || isContinuation) && !lastBotMessageFound) {
                    if (history[i].from === process.env.BOTNAME) {
                        lastBotMessageFound = true
                        if (isRetry) {
                            continue
                        }
                    } else {
                        continue
                    }
                }

                const promptHistoryLength = encoder.encode(promptHistory).length
                const line = (history[i].from ? `${history[i].from}: ${history[i].msg}` : history[i].msg) + '\n'
                const lineLength = encoder.encode(line).length
                if (contextLength + promptHistoryLength + lineLength + lastLineLength < (parseInt(process.env.TOKEN_LIMIT || "2048") - 152)) {
                    promptHistory = line + promptHistory
                } else {
                    couldInsertAllHistory = false
                    break
                }
            }
        }

        // General context
        // Bot context
        // Remembered things
        // Bot presentation message
        // ...
        // History messages
        let completePrompt = promptContext + (couldInsertAllHistory ? "" : "...\n") + promptHistory
        if (isContinuation) {
            completePrompt = completePrompt.substr(0, completePrompt.length - 1)
        } else {
            completePrompt += lastLine
        }
        const completePromptLength = encoder.encode(completePrompt).length
        return {prompt: completePrompt, repetition_penalty_range: completePromptLength - minimalContextLength}
    }
}

export default PromptService