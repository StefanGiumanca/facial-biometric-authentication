FROM node:20-alpine

WORKDIR /app

COPY web-admin/package*.json ./
RUN npm ci

COPY web-admin ./

EXPOSE 5173

CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0"]
