import {config} from "dotenv";

config()
import Command from "./Command.js";
import historyService from "../service/historyService.js";
import DanbooruService from "../externalApi/danbooruService.js";


const danbooruCommands = {
    danbooru: new Command(
        "Danbooru",
        [],
        ["!danbooru "],
        process.env.ALLOW_DANBOORU,
        async (msg, parsedMsg, from, channel, command, roles, messageId, targetMessageId, client, attachmentUrl) => {
            const result = await DanbooruService.getTags(parsedMsg || null)

            if (result) {
                historyService.pushIntoHistory(msg, from, channel, messageId)
                const formattedEvent = `[ ${process.env.BOTNAME} responds to the command by searching "${parsedMsg}" on the hentai website "danbooru" and sending one of the pictures to ${from}. The picture has the tags "${
                    result.tag_string_general
                        ?.split(',')
                        .slice(0,5)
                        .map(e=>e.trim())
                        .join(', ')
                }" ]`
                return {
                    message: `# Id: ${result?.id}\nTags_string_general: ${result.tag_string_general}\nTag_string_character: ${result.tag_string_character}\nArtist: ${result.tag_string_artist}\nDate: ${result.created_at}\nURL: ||${result.large_file_url}||`,
                    success: true,
                    instantReply: true,
                    image: result.large_file_url,
                    pushIntoHistory: [formattedEvent, null, channel]
                }
            } else {
                return {
                    error: `# I'm sorry, but your search didn't return any result... Maybe try another keyword!`
                }
            }
        },
        false
    ),
}

danbooruCommands.all = [
    danbooruCommands.danbooru,
]

export default danbooruCommands