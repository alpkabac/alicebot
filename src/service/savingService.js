import {config} from "dotenv";
import personalityService from "./personalityService.js";
import historyService from "./historyService.js";
import memoryService from "./memoryService.js";
import utils from "../utils.js";
import fs from "fs";
import pawnService from "./rpg/pawnService.js";
import playerService from "./rpg/playerService.js";
import worldItemsService from "./rpg/worldItemsService.js";

config()

class SavingService {

    static save(channel) {

        const personality = personalityService.getChannelPersonality(channel)

        // Prevents saving DMs
        if (channel.startsWith("##") && !personality?.enabledDMChannels?.[channel]) return false

        const data = this.getCompleteJSON(channel)

        try {
            fs.mkdirSync(`./bot/${process.env.BOT_ID}/save`)
        } catch (e) {

        }

        try {
            const filename = `./bot/${process.env.BOT_ID}/save/${channel}.json`
            utils.save(filename, JSON.stringify(data, null, 4))
            return true
        } catch (e) {
            console.log(e)
        }
    }

    /**
     *
     * @param channel
     * @return {{memory: {}, personality, activeItems, players: {}, lastPawnCreatedAt: null, history: [], pawn: *}}
     */
    static getCompleteJSON(channel) {
        const personality = personalityService.getChannelPersonality(channel)
        const history = historyService.getChannelHistory(channel)
        const memory = memoryService.getChannelMemory(channel)
        const pawn = pawnService.getActivePawn(channel)
        const lastPawnCreatedAt = pawnService.lastPawnCreatedAt[channel] || null
        const players = playerService.players[channel] || {}
        const activeItems = worldItemsService.getActiveItems(channel)

        return {
            personality,
            history,
            memory,
            pawn,
            lastPawnCreatedAt,
            players,
            activeItems
        }
    }

    static loadJSON(channel, json) {
        try {
            const {personality, history, memory, pawn, lastPawnCreatedAt, players, activeItems} = json
            if (personality) {
                personalityService.channelBotPersonality[channel] = personality
            }

            if (history) {
                historyService.channelHistories[channel] = history
            }

            if (memory) {
                memoryService.channelMemories[channel] = memory
            }

            if (pawn) {
                pawnService.activePawn[channel] = pawn
            }

            if (lastPawnCreatedAt) {
                pawnService.lastPawnCreatedAt[channel] = lastPawnCreatedAt
            }

            if (players) {
                playerService.players[channel] = players
            }

            if (activeItems) {
                worldItemsService.activeItems[channel] = activeItems
            }

            if (personality || history || memory)
                return true
        } catch (e) {
            console.log(e)
        }

        return false
    }

    static load(channel) {
        try {
            const json = utils.loadJSONFile(`./bot/${process.env.BOT_ID}/save/${channel}.json`)
            this.loadJSON(channel, json)
        } catch (e) {
            console.log(e)
        }

        return false
    }

    static loadAllChannels() {
        try {
            const channelFiles = fs.readdirSync(`./bot/${process.env.BOT_ID}/save`)
            if (channelFiles) {
                for (let channelFile of channelFiles) {
                    SavingService.load(channelFile.replace('.json', ''))
                }
            }
        } catch (e) {

        }
    }

}

export default SavingService