FROM node:20-alpine

WORKDIR /app

COPY visionauth-mobile/package*.json ./
RUN npm ci

COPY visionauth-mobile ./

ENV EXPO_NO_TELEMETRY=1
ENV EXPO_DEVTOOLS_LISTEN_ADDRESS=0.0.0.0

EXPOSE 8081 19000 19001 19002

CMD ["npm", "run", "start", "--", "--clear", "--lan"]
