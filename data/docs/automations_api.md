# Nimbus Automations and Local API

## Creating automations
Automations let devices react to each other. Open the app, tap Automations and choose a trigger such as motion detected, a schedule, or sunset. Then pick one or more actions, for example turn on a light or lock a door. You can create up to 100 automations per hub. Automations keep running locally even when the internet is down.

## Scenes
A scene stores the state of several devices so you can recall them together, such as a Movie Night scene that dims the lights and closes the blinds. Scenes can be triggered from the app, from a physical button, or by a voice command to the hub.

## Voice commands
Say the wake word Hey Nimbus followed by a command, for example Hey Nimbus turn off the kitchen lights. The hub understands English, Spanish and German. You can change the wake word to Computer or Jarvis in Settings.

## Local REST API
The hub exposes a local REST API on port 8443 for developers. Enable it in the app under Settings, then Developer, which generates an API token. Send the token in an Authorization Bearer header with each request. The API allows 60 requests per minute, and requests over the limit receive an HTTP 429 response. The API is only reachable from your home network.
